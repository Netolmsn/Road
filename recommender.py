import os
import osmnx as ox
import networkx as nx
import folium
import numpy as np
import time

# --- Configurações Globais ---
# Use 'drive' para carros. 'walk' e 'bike' também são opções.
NETWORK_TYPE = 'drive' 

# Configurações de Custo (Exemplos - ajuste para sua realidade)
FUEL_PRICE_PER_LITER = 6.19 # Preço em Reais
VEHICLE_CONSUMPTION_L_PER_KM = 1 / 12 # Veículo faz 12 km/L

# Arquivo para salvar o grafo (evita baixar toda vez)
GRAPH_FILE = "rio_de_janeiro_brasil.graphml" # <--- ALTERADO


def get_graph(place_name):
    """
    Baixa o grafo do OpenStreetMap (OSM) se não existir localmente,
    ou carrega do arquivo .graphml.
    """
    if os.path.exists(GRAPH_FILE):
        print(f"Carregando grafo de '{GRAPH_FILE}'...")
        G = ox.load_graphml(GRAPH_FILE)
    else:
        print(f"Baixando dados de '{place_name}' do OSM...")
        G = ox.graph_from_place(place_name, network_type=NETWORK_TYPE)
        
        print("Enriquecendo arestas (métricas)...")
        G = enrich_edges(G)
        
        print(f"Salvando grafo em '{GRAPH_FILE}'...")
        ox.save_graphml(G, GRAPH_FILE)
    
    return G

def enrich_edges(G):
    """
    Adiciona as métricas de custo (tempo, custo, estética)
    a cada aresta (segmento de rua) no grafo.
    """
    # 1. TEMPO: OSMnx tem funções prontas para isso
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # 2. CUSTO e 3. ESTÉTICA
    for u, v, key, data in G.edges(keys=True, data=True):
        
        # --- CÁLCULO DE CUSTO ---
        distance_km = data['length'] / 1000
        fuel_cost = distance_km * VEHICLE_CONSUMPTION_L_PER_KM * FUEL_PRICE_PER_LITER
        toll_cost = 12.0 if data.get('toll', 'no') == 'yes' else 0 # Proxy de pedágio em R$
        data['cost'] = fuel_cost + toll_cost

        
        # --- CÁLCULO DE ESTÉTICA (Proxy V1) ---
        score = 5.0 # Nota base (neutra)
        highway_type = data.get('highway', '')
        
        if highway_type == 'motorway': # Ex: Linha Vermelha, Linha Amarela
            score = 1.0 
        elif highway_type in ['primary', 'secondary']: # Ex: Av. Atlântica, Av. Brasil
            score = 6.0
        elif highway_type in ['residential', 'tertiary']: # Ruas de bairro
            score = 8.0 
        
        if 'tunnel' in data: # Ex: Túnel Rebouças
            score = 0.5 
        
        # O proxy V1 não captura "vista para o mar". 
        # A Av. Atlântica (bonita) pode ter score similar à Av. Brasil (feia).
        # Isso é uma limitação para melhorarmos depois.
        
        data['aesthetic_score'] = score
        data['aesthetic_weight'] = (1 / (score + 1e-6)) * data['length']
        
    return G

def calculate_path_stats(G, path):
    """
    Calcula as métricas totais (tempo, custo, etc.) para um
    caminho (lista de nós).
    """
    # Alternativa mais moderna para calcular estatísticas de rota
route_gdfs = ox.utils_graph.route_to_gdfs(G, path)
edges = route_gdfs[1] # [1] é o GeoDataFrame de arestas
    
    total_length = sum(edge['length'] for edge in edges)
    total_time_s = sum(edge['travel_time'] for edge in edges)
    total_cost = sum(edge['cost'] for edge in edges)
    
    total_aesthetic_score = sum(edge['aesthetic_score'] * edge['length'] for edge in edges)
    avg_aesthetic_score = total_aesthetic_score / total_length if total_length > 0 else 0
    
    return {
        "distancia_km": total_length / 1000,
        "tempo_min": total_time_s / 60,
        "custo_total": total_cost,
        "estetica_media": avg_aesthetic_score
    }

def find_routes(G, start_coords, end_coords):
    """
    Encontra os 3 tipos de rotas "puras" (rápida, barata, bonita).
    """
    orig_node = ox.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
    dest_node = ox.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])

    routes = {}

    # --- 1. Rota +RÁPIDA ---
    print("Calculando Rota +Rápida...")
    routes['rapida'] = {
        'path': nx.shortest_path(G, orig_node, dest_node, weight='travel_time', method='dijkstra')
    }

    # --- 2. Rota +BARATA ---
    print("Calculando Rota +Barata...")
    routes['barata'] = {
        'path': nx.shortest_path(G, orig_node, dest_node, weight='cost', method='dijkstra')
    }
    
    # --- 3. Rota +BONITA ---
    print("Calculando Rota +Bonita...")
    routes['bonita'] = {
        'path': nx.shortest_path(G, orig_node, dest_node, weight='aesthetic_weight', method='dijkstra')
    }

    # Calcula as estatísticas para cada rota encontrada
    for route_type, data in routes.items():
        stats = calculate_path_stats(G, data['path'])
        data['stats'] = stats
        print(f"\n--- Estatísticas (Rota {route_type.upper()}) ---")
        print(f"  Distância: {stats['distancia_km']:.2f} km")
        print(f"  Tempo: {stats['tempo_min']:.2f} min")
        print(f"  Custo: R$ {stats['custo_total']:.2f}") 
        print(f"  Estética (Média): {stats['estetica_media']:.2f} / 10")

    return routes, (start_coords, end_coords)

def plot_map(G, routes_dict, coords):
    """
    Plota as 3 rotas em um mapa interativo Folium e o salva.
    """
    start_coords, end_coords = coords
    
    map_center = start_coords
    m = folium.Map(location=map_center, zoom_start=13) # <--- ALTERADO: Zoom 13 fica bom para o Rio

    # Cores para as rotas
    colors = {
        "rapida": "#FF0000", # Vermelho
        "barata": "#00FF00", # Verde
        "bonita": "#0000FF"  # Azul
    }

    for route_type, data in routes_dict.items():
        stats = data['stats']
        tooltip = f"""
            <b>Rota: {route_type.upper()}</b><br>
            Tempo: {stats['tempo_min']:.1f} min<br>
            Custo: R$ {stats['custo_total']:.2f}<br> 
            Estética: {stats['estetica_media']:.1f}
        """
        
        ox.plot_route_folium(
            G, 
            data['path'], 
            route_map=m, 
            color=colors[route_type], 
            weight=5, 
            opacity=0.7, 
            tooltip=tooltip
        )

    # Marcadores de Início e Fim
    folium.Marker(location=start_coords, popup="Origem", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=end_coords, popup="Destino", icon=folium.Icon(color='red')).add_to(m)

    output_file = "mapa_rotas.html"
    m.save(output_file)
    print(f"\nMapa salvo em '{output_file}'!")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    
    # 1. ESCOLHA DO LOCAL
    place = "Rio de Janeiro, Brasil" # <--- ALTERADO
    
    # 2. ESCOLHA DOS PONTOS
    # (Lat, Lon)
    # Ex: Do Aeroporto Santos Dumont (Centro) até Copacabana
    start_point = (-22.9103, -43.1631) # Aeroporto Santos Dumont (SDU)
    end_point = (-22.9839, -43.1931)   # Copacabana (Posto 5 / Rua Sá Ferreira)
    # <--- ALTERADO: Coordenadas dentro do Rio

    # 3. EXECUTAR O PROCESSO
    start_time = time.time()
    
    # Delete o arquivo .graphml anterior (de SP ou Porto) se ele existir
    # para forçar o download do novo mapa.
    graph = get_graph(place) 
    
    routes, coords = find_routes(graph, start_point, end_point)
    
    plot_map(graph, routes, coords)
    
    end_time = time.time()
    print(f"Processo concluído em {end_time - start_time:.2f} segundos.")