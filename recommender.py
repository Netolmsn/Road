import os
import osmnx as ox
import networkx as nx
import folium
import numpy as np
import time
import argparse # Import do CLI

# --- Importações Geo (V2) ---
try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import transform, unary_union # unary_union pode ser usado como fallback se union_all não existir
    import pyproj
except ImportError:
    print("Erro: Bibliotecas 'pandas', 'geopandas', 'shapely' ou 'pyproj' não encontradas.")
    print("Certifique-se de estar no ambiente Conda correto e ter rodado:")
    print("conda install -c conda-forge geopandas shapely pyproj pandas osmnx folium")
    exit()

# --- Configurações Globais ---
NETWORK_TYPE = 'drive'
FUEL_PRICE_PER_LITER = 6.19
VEHICLE_CONSUMPTION_L_PER_KM = 1 / 12
# O nome do arquivo do grafo agora é dinâmico (calculado no get_graph)

#
# ===================================================================
# >>> FUNÇÃO 'enrich_edges_v2' (Score Estético V2) CORRIGIDA <<<
# ===================================================================
#
def enrich_edges_v2(G, place_name):
    """
    Adiciona métricas de custo, tempo e o NOVO score estético (V2).
    """
    print("Iniciando enriquecimento V2...")
    # Garante que os atributos básicos de tempo existam
    if not nx.get_edge_attributes(G, "speed_kph"):
        G = ox.add_edge_speeds(G)
    if not nx.get_edge_attributes(G, "travel_time"):
        G = ox.add_edge_travel_times(G)

    print("Baixando POIs 'bonitos' (parques, água, mirantes)...")
    tags = {'leisure': 'park',
            'natural': ['water', 'coastline', 'beach'],
            'tourism': 'viewpoint'}
    beautiful_geometries_utm = None
    wgs84_to_utm = None
    try:
        # --- CORREÇÃO APLICADA AQUI ---
        pois_gdf = ox.features.features_from_place(place_name, tags=tags)
        # --- FIM DA CORREÇÃO ---

        if pois_gdf.empty:
             print("Aviso: Nenhum POI encontrado. Usando apenas score V1.")
        else:
            utm_crs = pois_gdf.estimate_utm_crs(datum_name="WGS 84")
            pois_gdf_utm = pois_gdf.to_crs(utm_crs)
            # Tenta usar union_all(), se não existir (versões antigas), usa unary_union
            try:
                beautiful_geometries_utm = pois_gdf_utm.union_all()
            except AttributeError:
                print("Aviso: 'union_all()' não disponível, usando 'unary_union'.")
                beautiful_geometries_utm = pois_gdf_utm.unary_union # Fallback
            print(f"POIs baixados e unificados. Usando CRS {utm_crs.to_string()}.")
            wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform

    except Exception as e:
        print(f"Aviso: Falha ao baixar/processar POIs. Usando apenas score V1. Erro: {type(e).__name__} - {e}")

    print("Calculando scores V1 (rua) + V2 (POI) para cada aresta...")
    edge_count = G.number_of_edges()
    processed_count = 0
    start_enrich_time = time.time()

    # Itera sobre uma cópia dos dados das arestas para evitar problemas de modificação durante a iteração
    edges_data = list(G.edges(keys=True, data=True))

    for u, v, key, data in edges_data:
        processed_count += 1
        if processed_count % 10000 == 0: # Imprime progresso
            elapsed = time.time() - start_enrich_time
            print(f"  Processando aresta {processed_count}/{edge_count} ({elapsed:.1f}s)...")

        # --- Custo (Garantir 'length') ---
        if 'length' not in data or not isinstance(data['length'], (int, float)):
             try: # Tenta recalcular se faltar
                 node_u = G.nodes[u]
                 node_v = G.nodes[v]
                 data['length'] = ox.distance.great_circle_vec(node_u['y'], node_u['x'], node_v['y'], node_v['x'])
             except Exception: data['length'] = 100 # Fallback extremo se nós também não tiverem coords
        distance_km = data['length'] / 1000
        fuel_cost = distance_km * VEHICLE_CONSUMPTION_L_PER_KM * FUEL_PRICE_PER_LITER
        toll_cost = 12.0 if data.get('toll', 'no') == 'yes' else 0
        data['cost'] = fuel_cost + toll_cost

        # --- Score V1 ---
        score_v1 = 5.0
        highway_type = data.get('highway', '')
        if highway_type == 'motorway': score_v1 = 1.0
        elif highway_type in ['primary', 'secondary']: score_v1 = 6.0
        elif highway_type in ['residential', 'tertiary']: score_v1 = 8.0
        if data.get('tunnel') == 'yes': score_v1 = 0.5

        # --- Score V2 Bonus ---
        beauty_bonus = 0.0
        if beautiful_geometries_utm and wgs84_to_utm:
            try:
                # Checa se nós têm coordenadas válidas
                node_u_data = G.nodes[u]
                node_v_data = G.nodes[v]
                if all(coord in node_u_data and coord in node_v_data and isinstance(node_u_data[coord], (int, float)) and isinstance(node_v_data[coord], (int, float)) for coord in ('x', 'y')):
                    edge_line_wgs84 = LineString([(node_u_data['x'], node_u_data['y']), (node_v_data['x'], node_v_data['y'])])
                    edge_line_utm = transform(wgs84_to_utm, edge_line_wgs84)
                    distance_m = edge_line_utm.distance(beautiful_geometries_utm)
                    # Garante que distance_m é um float
                    if isinstance(distance_m, (int, float)):
                         beauty_bonus = 10 * np.exp(-distance_m / 500)
            except Exception: # Ignora erros individuais no cálculo de bônus silenciosamente
                 pass

        data['aesthetic_score'] = score_v1 + beauty_bonus
        # Garante que 'length' existe antes de calcular weight
        data['aesthetic_weight'] = (1 / (data['aesthetic_score'] + 1e-9)) * data.get('length', 100) # Usa default length

        # Atualiza os dados da aresta no grafo original
        G[u][v][key].update(data)


    print("Enriquecimento V2 concluído.")
    return G

#
# ===================================================================
# >>> FUNÇÃO 'get_graph' (Nome de arquivo dinâmico e robusto) <<<
# ===================================================================
#
def get_graph(place_name):
    """
    Baixa ou carrega o grafo.
    """
    # Limpa o nome do lugar para criar um nome de arquivo seguro
    place_filename = "".join(c for c in place_name.lower() if c.isalnum() or c in (' ', '_', '-')).rstrip()
    GRAPH_FILE = f"{place_filename.replace(' ','_')}_v2.graphml"

    G = None # Inicializa G como None
    if os.path.exists(GRAPH_FILE):
        print(f"Carregando grafo V2 de '{GRAPH_FILE}'...")
        try:
            G = ox.load_graphml(GRAPH_FILE)
            print("Convertendo tipos de dados das arestas...")
            needs_resave = False # Flag para indicar se precisamos salvar novamente
            for u, v, key, data in G.edges(keys=True, data=True):
                try:
                    data['cost'] = float(data.get('cost', 0))
                    data['aesthetic_weight'] = float(data.get('aesthetic_weight', 1e9))
                    data['aesthetic_score'] = float(data.get('aesthetic_score', 0))
                    # Se travel_time não for float, tenta converter
                    if 'travel_time' not in data or not isinstance(data['travel_time'], (int, float)):
                        data['travel_time'] = float(data.get('length', 100) / 5) # Fallback tempo
                        needs_resave = True
                    # Garante que length é float
                    if 'length' not in data or not isinstance(data['length'], (int, float)):
                         data['length'] = 100.0 # Fallback length
                         needs_resave = True

                except (ValueError, TypeError) as conv_err:
                     print(f"  Aviso: Erro ao converter atributo para aresta ({u},{v},{key}): {conv_err}. Usando defaults.")
                     data['cost'] = data.get('cost', 0)
                     data['aesthetic_weight'] = data.get('aesthetic_weight', 1e9)
                     data['aesthetic_score'] = data.get('aesthetic_score', 0)
                     data['travel_time'] = data.get('travel_time', 20.0) # Default 20s
                     data['length'] = data.get('length', 100.0)
                     needs_resave = True # Marca para salvar com tipos corrigidos
            if needs_resave:
                print("Tipos de dados corrigidos no grafo carregado. Re-salvando...")
                try: ox.save_graphml(G, filepath=GRAPH_FILE)
                except Exception as save_err: print(f"Erro ao re-salvar grafo corrigido: {save_err}")

        except Exception as load_err:
             print(f"Erro ao carregar o arquivo '{GRAPH_FILE}': {load_err}")
             print("O arquivo pode estar corrompido. Tentando baixar novamente...")
             G = None # Reseta G para forçar o download

    # Se G ainda for None (falha no carregamento ou arquivo não existe)
    if G is None:
        print(f"Baixando dados de '{place_name}' do OSM...")
        ox.settings.timeout = 300 # Aumenta timeout para download
        try:
            # Simplifica o grafo e garante consistência bidirecional
            G_temp = ox.graph_from_place(place_name, network_type=NETWORK_TYPE, simplify=False)
            G_temp = ox.simplification.simplify_graph(G_temp)
            G = ox.utils_graph.get_largest_component(G_temp, strongly=True) # Pega componente principal
            print(f"Grafo baixado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
        except Exception as e:
            print(f"Erro Crítico ao baixar o grafo para '{place_name}': {type(e).__name__} - {e}")
            print("Verifique se o nome do lugar está correto (ex: 'Cidade, Estado, País') e tente novamente.")
            exit()

        print(f"Iniciando enriquecimento V2 para '{place_name}' (PODE LEVAR VÁRIOS MINUTOS)...")
        try:
            G = enrich_edges_v2(G, place_name)
        except Exception as enrich_err:
             print(f"Erro durante o enriquecimento V2: {enrich_err}")
             print("Continuando com os dados disponíveis...") # Tenta continuar mesmo se V2 falhar parcialmente

        print(f"Salvando grafo V2 enriquecido em '{GRAPH_FILE}'...")
        try:
            ox.save_graphml(G, filepath=GRAPH_FILE)
        except Exception as e:
            print(f"Erro ao salvar o grafo V2: {e}")

    print("Normalizando pesos das arestas (0-1)...")
    G = normalize_edge_weights(G) # Normaliza em ambos os casos (load/download)
    return G

#
# ===================================================================
# >>> RESTO DO SCRIPT (TOPSIS, ROTEAMENTO, PLOTAGEM) <<<
# ===================================================================
#
def rank_routes_topsis_manual(routes_dict, topsis_weights):
    print("\n" + "="*30)
    print("INICIANDO RANKING TOPSIS (Manual)")
    print("="*30)
    data = []
    route_names = []
    valid_routes_for_ranking = {name: data for name, data in routes_dict.items() if 'stats' in data and not any(np.isnan(val) for val in data['stats'].values())}

    if not valid_routes_for_ranking:
        print("Nenhuma rota com estatísticas válidas para rankear.")
        return None, []

    for route_name, route_data in valid_routes_for_ranking.items():
        stats = route_data['stats']
        route_names.append(route_name)
        data.append({
            'tempo_min': stats['tempo_min'],
            'custo_total': stats['custo_total'],
            'estetica_media': stats['estetica_media']
        })
    matrix_df = pd.DataFrame(data, index=route_names)

    if matrix_df.isnull().values.any():
        print("Aviso: Valores NaN encontrados na matriz de decisão. Preenchendo com a média da coluna.")
        try:
            matrix_df = matrix_df.fillna(matrix_df.mean())
        except Exception as fill_err:
             print(f"Erro ao preencher NaNs: {fill_err}. Removendo linhas com NaN.")
             matrix_df = matrix_df.dropna()

        if matrix_df.isnull().values.any():
             print("Erro Fatal: Não foi possível preencher/remover todos os NaNs. Saindo do ranking.")
             return None, []
        if matrix_df.empty:
            print("Erro: Nenhuma rota válida restante após remover NaNs.")
            return None, []
        # Atualiza route_names se linhas foram removidas
        route_names = matrix_df.index.tolist()

    matrix = matrix_df.values
    objectives = np.array([-1, -1, 1])
    weights = np.array([topsis_weights['time'], topsis_weights['cost'], topsis_weights['beauty']])
    print(f"Critérios: [Tempo (MIN), Custo (MIN), Estética (MAX)]")
    print(f"Pesos Finais: {weights}")
    print("Alternativas (Rotas):")
    print(matrix_df)

    # --- Matemática TOPSIS ---
    norm_denominator = np.linalg.norm(matrix, axis=0)
    norm_denominator[norm_denominator < 1e-9] = 1e-9
    norm_matrix = matrix / norm_denominator
    weighted_matrix = norm_matrix * weights
    pis = np.where(objectives == 1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    nis = np.where(objectives == 1, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    dist_to_pis = np.linalg.norm(weighted_matrix - pis, axis=1)
    dist_to_nis = np.linalg.norm(weighted_matrix - nis, axis=1)
    closeness_denominator = dist_to_pis + dist_to_nis
    closeness_denominator[closeness_denominator < 1e-9] = 1e-9
    closeness_coefficient = dist_to_nis / closeness_denominator
    # Garante que ranks usem o tamanho correto da matriz após possível remoção de NaNs
    ranks = np.argsort(closeness_coefficient)[::-1]
    # --- Fim ---

    print("\n--- RANKING FINAL (TOPSIS Manual) ---")
    result_df = matrix_df.copy()
    result_df['Score (Ci)'] = closeness_coefficient
    # Usa os nomes do DataFrame (que podem ter sido filtrados)
    ranked_routes = result_df.index[ranks].tolist()
    result_df = result_df.reindex(ranked_routes)
    result_df['Rank'] = range(1, len(result_df) + 1)
    print(result_df)

    if not ranked_routes:
        print("Erro: Ranking não produziu resultados.")
        return None, []

    best_route_name = ranked_routes[0]
    print(f"\n🏆 A Rota #1 (Melhor Trade-Off) é: '{best_route_name}'")
    return best_route_name, ranks


def normalize_edge_weights(G):
    valid_edge_data = []
    # Itera de forma segura, checando se 'data' existe
    for u, v, data in G.edges(data=True):
         if data is not None and all(k in data and isinstance(data[k], (int, float)) and not np.isnan(data[k])
                for k in ['travel_time', 'cost', 'aesthetic_weight']):
             valid_edge_data.append(data)

    if not valid_edge_data:
        print("Aviso: Nenhuma aresta com dados numéricos completos para normalização.")
        for u, v, key, data in G.edges(keys=True, data=True):
            if data is not None: # Verifica se data não é None
                data['norm_time'] = 1.0
                data['norm_cost'] = 1.0
                data['norm_beauty'] = 1.0
        return G

    times = [d['travel_time'] for d in valid_edge_data]
    costs = [d['cost'] for d in valid_edge_data]
    beauties = [d['aesthetic_weight'] for d in valid_edge_data]

    if not times or not costs or not beauties: # Segurança extra
        print("Aviso: Listas de métricas vazias. Pulando normalização.")
        return G

    min_time, max_time = min(times), max(times)
    range_time = max_time - min_time + 1e-9
    min_cost, max_cost = min(costs), max(costs)
    range_cost = max_cost - min_cost + 1e-9
    min_beauty, max_beauty = min(beauties), max(beauties)
    range_beauty = max_beauty - min_beauty + 1e-9

    for u, v, key, data in G.edges(keys=True, data=True):
        if data is not None: # Verifica se data não é None
            if all(k in data and isinstance(data[k], (int, float)) and not np.isnan(data[k])
                   for k in ['travel_time', 'cost', 'aesthetic_weight']):
                data['norm_time'] = max(0, min(1, (data['travel_time'] - min_time) / range_time))
                data['norm_cost'] = max(0, min(1, (data['cost'] - min_cost) / range_cost))
                data['norm_beauty'] = max(0, min(1, (data['aesthetic_weight'] - min_beauty) / range_beauty))
            else:
                data['norm_time'] = 1.0
                data['norm_cost'] = 1.0
                data['norm_beauty'] = 1.0
    print("Normalização concluída.")
    return G


def add_composite_weight(G, w_time, w_cost, w_beauty):
    for u, v, key, data in G.edges(keys=True, data=True):
        if data is not None: # Verifica se data não é None
            data['composite'] = (
                w_time * data.get('norm_time', 1.0) +
                w_cost * data.get('norm_cost', 1.0) +
                w_beauty * data.get('norm_beauty', 1.0)
            )
    return G

def calculate_path_stats(G, path):
    stats_nan = {"distancia_km": np.nan, "tempo_min": np.nan, "custo_total": np.nan, "estetica_media": np.nan}
    if not path or not isinstance(path, (list, tuple)) or len(path) < 2:
         return stats_nan
    try:
        edge_attributes = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not G.has_edge(u, v):
                 # print(f"Aviso: Aresta ({u}, {v}) não encontrada no grafo durante cálculo de stats.")
                 continue # Pula aresta inválida
            # Tenta pegar a aresta com key 0, senão a primeira que encontrar
            edge_data = G.get_edge_data(u, v, key=0)
            if edge_data is None:
                 edge_data_dict = G.get_edge_data(u,v)
                 if edge_data_dict: edge_data = next(iter(edge_data_dict.values()))
                 else: continue # Pula se não encontrar nenhuma aresta

            edge_attributes.append(edge_data)

        if not edge_attributes:
             # print(f"Aviso: Não foi possível obter atributos de aresta válidos para o caminho {path[:3]}...")
             return stats_nan

        edges = pd.DataFrame(edge_attributes)
        required_cols = ['length', 'travel_time', 'cost', 'aesthetic_score']
        # Verifica e preenche colunas ausentes
        for col in required_cols:
             if col not in edges.columns:
                 edges[col] = 0 if col in ['length', 'travel_time', 'cost'] else np.nan

        # Converte colunas para numérico, tratando erros
        for col in required_cols:
            edges[col] = pd.to_numeric(edges[col], errors='coerce')

        # Recalcula NaN stats se a conversão falhou
        if edges[required_cols[:-1]].isnull().values.any():
             print("Aviso: NaN encontrado em colunas essenciais após conversão. Retornando stats NaN.")
             return stats_nan

        total_length = edges['length'].sum()
        total_time_s = edges['travel_time'].sum()
        total_cost = edges['cost'].sum()

        avg_aesthetic_score = np.nan
        if 'aesthetic_score' in edges.columns and not edges['aesthetic_score'].isnull().all():
             valid_aesthetics = edges.dropna(subset=['aesthetic_score', 'length'])
             if not valid_aesthetics.empty:
                  total_aesthetic_score = (valid_aesthetics['aesthetic_score'] * valid_aesthetics['length']).sum()
                  total_length_for_aesthetic = valid_aesthetics['length'].sum()
                  if total_length_for_aesthetic > 1e-9:
                      avg_aesthetic_score = total_aesthetic_score / total_length_for_aesthetic

        # Retorna NaN se algum cálculo essencial resultou em NaN
        final_stats = {
            "distancia_km": total_length / 1000,
            "tempo_min": total_time_s / 60,
            "custo_total": total_cost,
            "estetica_media": avg_aesthetic_score
        }
        if any(np.isnan(v) for k, v in final_stats.items() if k != 'estetica_media'): # Estética pode ser NaN
            # print("Aviso: NaN encontrado nos resultados finais de stats.")
            return stats_nan

        return final_stats
    except Exception as e:
        # print(f"Erro detalhado ao calcular estatísticas: {type(e).__name__} - {e}")
        return stats_nan


def find_routes(G, start_coords, end_coords, tradeoff_weights_list):
    try:
        orig_node = ox.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
        dest_node = ox.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
        print(f"Nó de origem encontrado: {orig_node}")
        print(f"Nó de destino encontrado: {dest_node}")
        # Verifica se origem e destino são o mesmo nó
        if orig_node == dest_node:
            print("Aviso: Nó de origem e destino são os mesmos.")
            # Retorna um dicionário com uma rota "vazia" ou informações mínimas
            return {'mesmo_ponto': {'path': [orig_node], 'stats': {'distancia_km': 0, 'tempo_min': 0, 'custo_total': 0, 'estetica_media': np.nan}}}, (start_coords, end_coords)

    except Exception as e:
        print(f"Erro Crítico: Não foi possível encontrar nós no grafo próximos aos pontos fornecidos.")
        # Tenta fornecer mais contexto sobre o grafo
        graph_name = G.graph.get('name', 'Área Desconhecida') if hasattr(G, 'graph') and G.graph else 'Área Desconhecida'
        print(f"  Origem: {start_coords}, Destino: {end_coords}, Mapa: {graph_name}")
        print(f"  Erro original: {e}")
        return {}, (start_coords, end_coords)

    routes = {}
    unique_paths = set()

    def add_route(route_name, weight_attr, method='dijkstra'):
        nonlocal routes, unique_paths # Permite modificar variáveis externas
        path = None # Inicializa path
        try:
            print(f"Calculando Rota '{route_name}' (peso: {weight_attr})...")
            edge_sample = next(iter(G.edges(data=True)), None)
            if edge_sample is None: print("Erro: Grafo sem arestas."); return
            # Checa se o atributo existe E não é NaN na amostra
            if weight_attr not in edge_sample[2] or pd.isna(edge_sample[2][weight_attr]):
                 print(f"Aviso: Atributo '{weight_attr}' ausente ou NaN na amostra. Pulando '{route_name}'.")
                 # print(f" Amostra: {edge_sample[2]}") # Debug
                 return

            # Verifica se os nós de origem/destino existem no grafo
            if orig_node not in G or dest_node not in G:
                 print(f"Erro: Nó de origem ({orig_node}) ou destino ({dest_node}) não encontrado no grafo principal. O grafo pode ser desconectado.")
                 return

            path = nx.shortest_path(G, orig_node, dest_node, weight=weight_attr, method=method)
            path_tuple = tuple(path)

            if not path or len(path) < 2: # Caminho deve ter pelo menos 2 nós
                print(f"Aviso: Caminho inválido (vazio ou único nó) retornado para rota '{route_name}'.")
                return

            if path_tuple not in unique_paths:
                # Calcula stats ANTES de adicionar, para garantir validade
                stats = calculate_path_stats(G, path)
                if stats and not any(np.isnan(val) for val in stats.values()):
                    routes[route_name] = {'path': path, 'stats': stats} # Adiciona stats aqui
                    unique_paths.add(path_tuple)
                    print(f"Rota '{route_name}' válida encontrada (nós: {len(path)}).")
                else:
                    print(f"Aviso: Rota '{route_name}' calculada, mas stats são inválidos/NaN. Descartando.")
            else:
                 print(f"Rota '{route_name}' (ou similar) já encontrada.")

        except nx.NetworkXNoPath:
            print(f"Aviso: Não foi encontrado caminho para '{route_name}' entre {orig_node} e {dest_node} (peso '{weight_attr}').")
        except KeyError as e:
            print(f"Erro de chave ao calcular '{route_name}' (nó {e} não encontrado?): {e}. Pulando.")
            # Se path foi calculado, tenta ver o nó problemático
            # if path: print(f" Path parcial: {path}")
        except Exception as e:
            print(f"Erro inesperado ao calcular rota '{route_name}': {type(e).__name__} - {e}. Pulando.")

    # Calcula as rotas
    add_route('rapida', 'travel_time')
    add_route('barata', 'cost')
    add_route('bonita', 'aesthetic_weight')

    print(f"Calculando Fronteira de Pareto simulada com {len(tradeoff_weights_list)} combinações...")
    for i, weights in enumerate(tradeoff_weights_list):
        w_t, w_c, w_b = weights
        # Recalcula peso composto a cada iteração
        G = add_composite_weight(G, w_time=w_t, w_cost=w_c, w_beauty=w_b)
        route_name = f"tradeoff_{w_t*100:.0f}T_{w_c*100:.0f}C_{w_b*100:.0f}B"
        add_route(route_name, 'composite')

    # Os stats já foram calculados e validados dentro de add_route
    print(f"\n--- {len(routes)} rotas únicas com estatísticas válidas encontradas ---")

    # Verifica se o dicionário `routes` não está vazio antes de retornar
    if not routes:
        print("Nenhuma rota válida foi calculada com sucesso após todas as tentativas.")

    return routes, (start_coords, end_coords)


def plot_map(G, routes_dict, coords, best_route_name):
    start_coords, end_coords = coords
    map_center = start_coords if isinstance(start_coords, tuple) and len(start_coords) == 2 else (-22.95, -43.2)
    zoom = 13 if isinstance(start_coords, tuple) else 11

    m = folium.Map(location=map_center, zoom_start=zoom, tiles="cartodbpositron")
    pure_colors = { "rapida": "#e41a1c", "barata": "#4daf4a", "bonita": "#377eb8" } # Colorblind friendly

    # Fallback se G for None
    if G is None:
        print("Aviso: Grafo não disponível para plotagem.")
        routes_dict = {} # Limpa rotas se grafo não existe
        best_route_name = None

    if not routes_dict:
        print("Nenhuma rota válida encontrada para plotar.")
        # Adiciona marcadores mesmo sem rotas
        try:
             if isinstance(start_coords, tuple): folium.Marker(location=start_coords, popup="Origem", icon=folium.Icon(color='green', icon='play')).add_to(m)
             if isinstance(end_coords, tuple): folium.Marker(location=end_coords, popup="Destino", icon=folium.Icon(color='red', icon='stop')).add_to(m)
        except Exception as e: print(f"Erro ao adicionar marcadores: {e}")
        output_file = "mapa_rotas_sem_caminho.html"
        m.save(output_file)
        print(f"\nMapa (sem rotas) salvo em '{output_file}'!")
        return

    # Helper para obter localizações de forma segura
    def get_locations(path_nodes, graph):
        locs = []
        if graph is None: return locs
        for node in path_nodes:
            node_data = graph.nodes.get(node)
            if node_data and 'x' in node_data and 'y' in node_data:
                locs.append((node_data['y'], node_data['x']))
        return locs

    # Desenha rotas de trade-off (cinza)
    for route_type, data in routes_dict.items():
        if route_type in pure_colors or route_type == best_route_name: continue
        try:
            locations = get_locations(data['path'], G)
            if locations:
                folium.PolyLine(locations=locations, color="#adb5bd", weight=3, opacity=0.6, tooltip=route_type).add_to(m)
        except Exception as e: print(f"Erro plot trade-off '{route_type}': {e}")

    # Desenha rotas puras
    for route_type in pure_colors:
        if route_type not in routes_dict: continue
        try:
            data = routes_dict[route_type]
            stats = data['stats']
            tooltip_html = f"<b>ROTA PURA: {route_type.upper()}</b><br>" \
                           f"T: {stats['tempo_min']:.1f}m | C: R${stats['custo_total']:.2f} | B: {stats['estetica_media']:.1f}" # Mais conciso
            locations = get_locations(data['path'], G)
            if locations:
                 folium.PolyLine(locations=locations, color=pure_colors[route_type], weight=5, opacity=0.9, tooltip=tooltip_html).add_to(m)
        except Exception as e: print(f"Erro plot pura '{route_type}': {e}")

    # Desenha a melhor rota
    if best_route_name and best_route_name in routes_dict:
        try:
            data = routes_dict[best_route_name]
            stats = data['stats']
            tooltip_html = f"<b><span style='color:orange;'>🏆 #1: {best_route_name.upper()}</span></b><br>" \
                           f"T: {stats['tempo_min']:.1f}m | C: R${stats['custo_total']:.2f} | B: {stats['estetica_media']:.1f}"
            locations = get_locations(data['path'], G)
            if locations:
                folium.PolyLine(locations=locations, color="#000000", weight=10, opacity=0.8).add_to(m)
                folium.PolyLine(locations=locations, color="#FFBF00", weight=7, opacity=1.0, tooltip=tooltip_html).add_to(m)
        except Exception as e: print(f"Erro plot melhor '{best_route_name}': {e}")

    # Adiciona marcadores de início e fim
    try:
        if isinstance(start_coords, tuple): folium.Marker(location=start_coords, popup="Origem", icon=folium.Icon(color='green', icon='play')).add_to(m)
        if isinstance(end_coords, tuple): folium.Marker(location=end_coords, popup="Destino", icon=folium.Icon(color='red', icon='stop')).add_to(m)
    except Exception as e: print(f"Erro marcadores: {e}")

    output_file = "mapa_rotas.html"
    try:
        m.save(output_file)
        print(f"\nMapa salvo em '{output_file}'!")
    except Exception as e: print(f"Erro ao salvar o mapa: {e}")


#
# ===================================================================
# >>> PONTO DE ENTRADA (CLI) - BLOCO CORRIGIDO <<<
# ===================================================================
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Recomendador de Rotas Multiobjetivo (V2 com CLI)")

    parser.add_argument('--orig', type=str, required=True,
                        help="Endereço de origem. Ex: 'Aeroporto Santos Dumont, Rio de Janeiro, Brasil'")
    parser.add_argument('--dest', type=str, required=True,
                        help="Endereço de destino. Ex: 'Cristo Redentor, Rio de Janeiro, Brasil'")
    parser.add_argument('--wt', type=float, default=0.3, help="Peso do Tempo (0.0-1.0)")
    parser.add_argument('--wc', type=float, default=0.2, help="Peso do Custo (0.0-1.0)")
    parser.add_argument('--wb', type=float, default=0.5, help="Peso da Beleza (0.0-1.0)")

    args = parser.parse_args()

    # --- ETAPA 1: Geocoding e Place Name ---
    print("Convertendo endereços para coordenadas (Geocoding)...")
    start_point = None
    end_point = None
    place_name = None
    try:
        start_point = ox.geocode(args.orig)
        end_point = ox.geocode(args.dest)

        print("Determinando o contexto do mapa (Usando ox.reverse_geocode)...")
        try:
             # --- CORREÇÃO APLICADA AQUI ---
             # Tenta usar ox.reverse_geocode (diretamente no ox)
             address = ox.reverse_geocode(start_point[0], start_point[1], language='pt')
             # --- FIM DA CORREÇÃO ---

             city = address.get('city', address.get('town', address.get('village')))
             state = address.get('state')
             country = address.get('country')
             place_parts = [part for part in [city, state, country] if part]

             if len(place_parts) >= 2:
                 place_name = ", ".join(place_parts)
             else: raise ValueError("ox.reverse_geocode não retornou info suficiente.") # Força fallback

        except AttributeError: # Captura erro se ox.reverse_geocode não existir
             print("Erro: 'ox.reverse_geocode' não encontrado. Tentando fallback...")
             raise ValueError("Fallback necessário") # Força fallback
        except Exception as reverse_err: # Captura outros erros do reverse_geocode
             print(f"Aviso: ox.reverse_geocode falhou ({reverse_err}). Usando fallback...")
             raise ValueError("Fallback necessário") # Força fallback

    except ValueError: # Entra aqui se o reverse_geocode falhou ou foi incompleto
        print("Executando fallback para extrair place_name da string de origem...")
        try:
            orig_parts = [part.strip() for part in args.orig.split(',')[-3:]]
            if len(orig_parts) >= 2:
                place_name = ", ".join(orig_parts[max(0, len(orig_parts)-2):])
            else:
                 orig_parts_fallback = args.orig.split()[-2:]
                 if len(orig_parts_fallback) >= 2:
                      place_name = " ".join(orig_parts_fallback)
                 else: raise ValueError("Falha no fallback. Não foi possível determinar o lugar.")
            # Garante que start/end point foram pegos
            if start_point is None: start_point = ox.geocode(args.orig)
            if end_point is None: end_point = ox.geocode(args.dest)
        except Exception as fallback_e:
              print(f"Erro Crítico no fallback de geocoding: {fallback_e}")
              exit()

    except Exception as e:
        print(f"Erro Crítico na Etapa 1 (Geocoding/Place Name): {type(e).__name__} - {e}")
        print("Verifique os endereços. Inclua Cidade, Estado, País.")
        exit()

    # Checagem final se tudo foi definido
    if start_point is None or end_point is None or place_name is None:
        print("Erro Crítico: Falha ao determinar pontos ou nome do local.")
        exit()

    print(f"Origem: {args.orig} -> {start_point}")
    print(f"Destino: {args.dest} -> {end_point}")
    print(f"Contexto do Mapa (Place): {place_name}")


    # --- ETAPA 2: Configurações ---
    pareto_combinations = [
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8),
        (0.33, 0.33, 0.34)
    ]
    topsis_user_preferences = {"time": args.wt, "cost": args.wc, "beauty": args.wb}

    # Validação e ajuste de pesos
    if not np.isclose(args.wt + args.wc + args.wb, 1.0):
        print("Aviso: Soma dos pesos != 1.0. Ajustando...")
        total_weight = args.wt + args.wc + args.wb
        if total_weight > 1e-9:
             topsis_user_preferences["time"] /= total_weight
             topsis_user_preferences["cost"] /= total_weight
             topsis_user_preferences["beauty"] /= total_weight
        else:
             print("Erro: Pesos zero. Usando padrão.")
             topsis_user_preferences = {"time": 0.33, "cost": 0.33, "beauty": 0.34}

    # --- Execução Principal ---
    start_time = time.time()
    best_route = None
    all_routes_dict = {}
    graph = None # Inicializa grafo como None
    coords = (start_point, end_point)

    try:
        graph = get_graph(place_name)
        if graph:
            # Verifica se os pontos estão dentro dos limites do grafo baixado
            # bounds = G.graph['streets_within_query_bbox'] # Precisa do bbox salvo no grafo
            # if not (bounds[0] <= start_point[0] <= bounds[2] and bounds[1] <= start_point[1] <= bounds[3]):
            #      print("Erro: Ponto de origem fora dos limites do mapa baixado.")
            # elif not (bounds[0] <= end_point[0] <= bounds[2] and bounds[1] <= end_point[1] <= bounds[3]):
            #      print("Erro: Ponto de destino fora dos limites do mapa baixado.")
            # else:
            all_routes_dict, coords = find_routes(graph, start_point, end_point, pareto_combinations)
            if all_routes_dict:
                 best_route, rank = rank_routes_topsis_manual(all_routes_dict, topsis_user_preferences)
            else: print("Nenhuma rota encontrada.")
        else: print("Falha ao carregar/criar o grafo.")

    except Exception as e:
        print(f"Erro Crítico durante a execução principal: {type(e).__name__} - {e}")
        # Tenta imprimir traceback para debug
        # import traceback
        # traceback.print_exc()


    # Plotagem
    plot_map(graph, all_routes_dict, coords, best_route)

    end_time = time.time()
    print(f"Processo concluído em {end_time - start_time:.2f} segundos.")