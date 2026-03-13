import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


############# 定义风险类别 ############
risk_category={
    "safe":0,
    "ghost":1,
    "collision":2
}

############# 定义节点编码逻辑-节点特征 #############
node_feature={
    "scenario": 0, "road_structure": 1, "road_facility": 2, "road_change": 3, "weather_environment": 4, "digital_information": 5,
    "traffic_participants": 6, "ego_information": 7,
    "frame0": 8, "frame1": 9, "frame2": 10, "frame3": 11, "frame4": 12, "frame5": 13, "frame6": 14, "frame7": 15, "frame8": 16, "frame9": 17,
    "road": 18, "lane": 19, "marker": 20, "stop sign": 21, "traffic light": 22, "barrier": 23, "area": 24, "road_defect": 25, "temporal_event": 26,
    "clutter": 27, "weather": 28, "uncommon": 29, "person": 30, "vehicle": 31, "ego_car": 32,
    "lane_1": 33, "lane_2": 34, "lane_3": 35, "lane_4": 36, "lane_5": 37,
    "weather-attr": 38,
    "ego": 39, "uncommon-s": 40, "person-s": 41, "vehicle-s": 42, "speed-attr": 43,
    "light-s": 44, "light-state": 45, "weather-s": 46,
    "speed": 47, "position": 48, "angle": 49,
    "bicycle": 50, "car": 51, "motorboke": 52, "bus": 53, "truck": 54, "stop sign-s": 55, "stop sign-state": 56,
    "cat": 57, "dog": 58,  "behavior_status": 59
}

############## 定义边类别编码逻辑-边特征 #################
edge_attr={
    "has": 0, "has_subclass": 1, "is_a": 2, "has_att": 3, "has_frame": 4, "in_next_frame": 5, "next_to": 6, "is_on": 7, "has_next_frame": 8,
    "red": 9, "yellow": 10, "green": 11,
    "Clear Night": 12, "Clear Noon": 13, "Clear Sunset": 14, "Cloudy Night": 15, "Cloudy Noon": 16, "Cloudy Sunset": 17, "Default": 18,
    "Dust Storm": 19, "Hard Rain Night": 20, "Hard Rain Noon": 21, "Hard Rain Sunset": 22, "Mid Rain Sunset": 23, "Mid Rainy Night": 24,
    "Mid Rainy Noon": 25, "Soft Rain Night": 26, "Soft Rain Noon": 27, "Soft Rain Sunset": 28, "Wet Cloudy Night": 29,
    "Wet Cloudy Noon": 30, "Wet Cloudy Sunset": 31, "Wet Night": 32, "Wet Noon": 33,

    "has effect": 34,

    "static": 35, "extremely-slow": 36, "very-slow": 37, "slow": 38, "low-normal": 39, "normal": 40, "high-normal": 41, "fast": 42,
    "very-fast": 43, "extremely-fast": 44, "dangerous-fast": 45,

    "angle-0-15": 46, "angle-15-25": 47, "angle-25-35": 48, "angle-35-45": 49, "angle-45-55": 50, "angle-55-65": 51,
    "angle-65-75": 52, "angle-75-80": 53, "angle-80-90": 54, "angle-90-180": 55,

    "dis-1": 56, "dis-2": 57, "dis-3": 58, "dis-4": 59, "dis-5": 60, "dis-6": 61, "dis-7": 62, "dis-8": 63, "dis-9": 64,

    "left_side":65,"right_side":66,"left_front":67,"right_front":68,"front":69,

    "uniform":70,"speed_up":71,"speed_down":72,"hash_slow":73,"hash_up":74
}


def graph_to_torch_data(G):
    """
    Convert NetworkX graph to PyTorch Geometric data format
    Args:
        G (networkx.DiGraph): The input knowledge graph.
    Returns:
        Data: The graph in PyTorch Geometric format.
    """
    # 获取所有节点
    node_list = list(G.nodes())

    # 构建邻接矩阵
    adj_matrix = nx.adjacency_matrix(G, nodelist=node_list).toarray()  # 转换为稀疏矩阵
    adj_matrix = sp.csr_matrix(adj_matrix)  # 生成稀疏矩阵

    # 创建节点特征矩阵，这里暂时用节点类别作为特征，后续可以改为其他特征
    node_features = np.array([G.nodes[node]["class_number"] for node in node_list]).reshape(-1, 1)

    # 转换为torch张量
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)


def export_triple(G):
    triplets = []
    # Iterate over edges to create the triplets (node1, relation, node2)
    for u, v, attr in G.edges(data=True):
        relation = attr.get('relation', 'unknown')  # Default to 'unknown' if no relation attribute
        triplet = (u, relation, v)
        triplets.append(triplet)

    return triplets


def export_node_feature_matrix(G,node_feature_number):
    num_nodes = G.number_of_nodes()  # 获取本G中的节点数量
    # print(num_nodes)
    # print(node_feature_number)
    node_feature_matrix = np.zeros([G.number_of_nodes(), node_feature_number])  # 搭建一个空表
    # 创建节点特征矩阵
    node_emnudict = {}
    edge_connection_index = []
    edge_type = []
    index = 0
    for i, node in enumerate(G.nodes):
        label = G.nodes[node].get('label', None)
        node_feature_matrix[i][node_feature[label]] = 1
        node_emnudict[node] = index
        index += 1

    # print(node_emnudict)
    for i in list(G.edges(data=True)):
        node1,node2,relation_dict = i
        edge_connection_index.append([node_emnudict[node1], node_emnudict[node2]])
        type_ = list(relation_dict.values())[0]
        edge_type.append(edge_attr[type_])
    # print(edge_connection_index)
    # print(edge_type)

    return node_feature_matrix,edge_connection_index,edge_type


def plot_AVKG_3D(G, image_size=(800, 600)):
    # Extract 3D positions and separate colors for nodes and edges
    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = []
    node_colors = []

    # Color mappings for classes
    class_color_map = {
        0: (200, 0, 0),      # Red
        1: (0, 0, 200),      # Blue
        2: (200, 200, 0),    # Yellow
        3: (0, 200, 0),      # Green
        4: (128, 0, 128)     # Purple
    }

    edge_color_map = {
        0: (0, 0, 255),      # Blue
        4: (100, 100, 100)
    }

    # Assign edge colors based on edge_class_number
    for u, v, attr in G.edges(data=True):
        edge_color = edge_color_map.get(attr['edge_class_number'], (0, 0, 0))
        edge_colors.append(tuple(c / 255 for c in edge_color) + (0.01,))

    # Assign node colors based on class_number
    for node, attr in G.nodes(data=True):
        node_color = class_color_map.get(attr['class_number'], (255, 255, 255))
        node_colors.append(tuple(c / 255 for c in node_color) + (0.9,))

    # 3D Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw nodes
    xs, ys, zs = zip(*[pos[node] for node in G.nodes()])
    ax.scatter(xs, ys, zs, c=node_colors, s=60, depthshade=True, edgecolors='k')

    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color=edge_colors[i], alpha=0.4)
        # Add edge labels (midpoint of the edge)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        mid_z = (pos[u][2] + pos[v][2]) / 2

        # Get the relation for the edge, assuming it's stored as 'relation' in edge attributes
        relation = G[u][v].get('relation', 'unknown')  # Default to 'unknown' if no relation attribute

        # Add text at the midpoint of the edge
        ax.text(mid_x, mid_y, mid_z, relation, fontsize=4, color='black', ha='center', va='center')


    # Add labels
    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, node, fontsize=5)

    # Plot a fixed Z plane (horizontal plane at Z_plane_height)
    x_range = np.linspace(min(xs), max(xs), 10)  # X-axis range
    y_range = np.linspace(min(ys), max(ys), 10)  # Y-axis range
    X, Y = np.meshgrid(x_range, y_range)  # Create a grid for the plane
    Z = np.full_like(X, 4800)  # Set the Z values to the fixed height

    # Plot the surface (plane)
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.2)

    ax.grid(None)

    ax.axis('on')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks
    # Set axis labels
    # ax.set_xlabel("X Axis")
    ax.set_ylabel("Time t")
    # ax.set_zlabel("Z Axis")
    plt.title("Temporal Knowledge Graph Visualization")
    plt.show()

    # Convert the Matplotlib figure to an OpenCV-compatible image (RGB format)
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Get the image as a numpy array (RGBA)
    img = np.array(canvas.buffer_rgba())

    # Convert from RGBA to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Resize the image to the desired output size (800x600)
    img_resized = cv2.resize(img, image_size)

    # Show the image with OpenCV
    cv2.imshow('3D Knowledge Graph', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_resized


def AVKG_3D_simple(scene, scene_len, kg_width=2400, kg_length=2400, kg_height=2400,is_visualization=False,is_export_triple=False,is_export_gnn=False):
    G = nx.DiGraph()
    relationship_describe = ["has", "has_subclass", "is_a", "has_att", "has_frame","in_next_frame","next_to","is_on","has_next_frame","has effect"]

    # Add scenario node
    G.add_node("scenario", label="scenario", class_number=0, pos=(kg_width / 2, kg_length/ 2, kg_height))
    # Add common node  "road_structure" 等普通的节点7层及其链接
    G.add_node("traffic_participants", label="traffic_participants", class_number=1, pos=(kg_width / 2-200, kg_length / 2, kg_height + 200))
    G.add_node("ego_information", label="ego_information", class_number=1,
               pos=(kg_width / 2 - 200, kg_length/ 4*3, kg_height + 200))
    G.add_node("road_facility", label="road_facility", class_number=1, pos=(kg_width / 2+200, kg_length/ 2, kg_height+200))
    G.add_node("road_change", label="road_change", class_number=1, pos=(kg_width / 2, kg_length / 2, kg_height + 200))
    G.add_node("road_structure", label="road_structure", class_number=1, pos=(kg_width / 2 - 100, kg_length / 4, kg_height + 200))
    G.add_node("weather_environment", label="weather_environment", class_number=1, pos=(kg_width / 2, kg_length/ 4, kg_height+200))
    G.add_node("digital_information", label="digital_information", class_number=1, pos=(kg_width / 2, kg_length/ 4*3, kg_height+200))
    G.add_edge("scenario", "road_structure", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "road_facility", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "road_change", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "weather_environment", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "digital_information", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "traffic_participants", relation=relationship_describe[0], edge_class_number=4)
    G.add_edge("scenario", "ego_information", relation=relationship_describe[0], edge_class_number=4)


    #####  添加第1层 road_structure  #####
    G.add_node("road", label="road", class_number=2, pos=(kg_width / 2 -150, kg_length / 4 -100, kg_height + 100))
    G.add_edge("road_structure", "road", relation=relationship_describe[1], edge_class_number=4)


    ###########  添加第2层 road_facility  #####
    G.add_node("stop sign", label="stop sign", class_number=2, pos=(kg_width / 2 +400, kg_length/ 6, kg_height+100))
    G.add_node("marker", label="marker", class_number=2, pos=(kg_width / 2 +400, kg_length / 6 *2, kg_height + 100))
    G.add_node("lane", label="lane", class_number=2, pos=(kg_width / 2 +400, kg_length / 6 * 3, kg_height + 100))
    G.add_node("traffic light", label="traffic light", class_number=2, pos=(kg_width / 2 +400, kg_length / 6 * 4, kg_height + 100))
    G.add_node("barrier", label="barrier", class_number=2, pos=(kg_width / 2 +400, kg_length / 6 * 5, kg_height + 100))
    G.add_edge("road_facility", "lane", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_facility", "marker", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_facility", "stop sign", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_facility", "traffic light", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_facility", "barrier", relation=relationship_describe[1], edge_class_number=4)

    # # # # ################### # # light 实体
    if scene[1][3][2] == "green":
        G.add_node("light-s", label="light-s", class_number=4, pos=(kg_width / 2 + 500, kg_length / 6*5, kg_height-50))
        G.add_edge("traffic light", "light-s", relation=relationship_describe[2], edge_class_number=4)
        G.add_node("light-state", label="light-state", class_number=3, pos=(kg_width / 2 + 500, kg_length / 6*5, kg_height-350))
        G.add_edge("light-s","light-state", relation=scene[1][3][2], edge_class_number=4)
    #

    # lane下的实体及其关系
    for i, item in enumerate(scene[1][0]):
        # print(i)
        # print(item)
        G.add_node( item[0], label=item[0], class_number=4, pos=(kg_width / 2 +500, kg_length / 6 * 3 -600 +600*i, kg_height-50))
        G.add_edge( item[0], "lane", relation=relationship_describe[2], edge_class_number=4)
        if i>=1:
            G.add_edge(item[0], item[1], relation=relationship_describe[6], edge_class_number=4)


    #############  添加第3层 road_change  #####
    G.add_node("area", label="area", class_number=2, pos=(kg_width / 2 -100, kg_length / 2, kg_height + 350))
    G.add_node("road_defect", label="road_defect", class_number=2, pos=(kg_width / 2, kg_length / 2, kg_height + 350))
    G.add_node("temporal_event", label="temporal_event", class_number=2, pos=(kg_width / 2 +100, kg_length / 2, kg_height + 350))
    G.add_node("clutter", label="clutter", class_number=2, pos=(kg_width / 2, kg_length / 2+400, kg_height + 350))
    G.add_edge("road_change", "area", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_change", "road_defect", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_change", "temporal_event", relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("road_change", "clutter", relation=relationship_describe[1], edge_class_number=4)


    #############  添加第4层  traffic_participants  #####
    # 添加 uncommon  vehicle  person等子类别
    G.add_node("uncommon", label="uncommon", class_number=2,
               pos=(kg_width / 2-250, kg_length / 6 * 3 -750, kg_height + 50))
    G.add_node("person", label="person", class_number=2,
               pos=(kg_width / 2-250, kg_length / 6*3, kg_height + 50))
    G.add_node("vehicle", label="vehicle", class_number=2,
               pos=(kg_width / 2-250, kg_length / 6 * 3 +750, kg_height + 50))

    G.add_edge("traffic_participants", "uncommon",relation=relationship_describe[1], edge_class_number=4)  # 关系是  subclass of
    G.add_edge("traffic_participants", "person",relation=relationship_describe[1], edge_class_number=4)
    G.add_edge("traffic_participants", "vehicle",relation=relationship_describe[1], edge_class_number=4)


    #############  添加第5层 weather_environment  #####
    G.add_node("weather", label="weather", class_number=2, pos=(kg_width / 2 + 200, kg_length/ 4 -500, kg_height+100))
    G.add_edge("weather_environment", "weather", relation=relationship_describe[1], edge_class_number=4)
    # 添加天气实体
    G.add_node("weather-s", label="weather-s", class_number=4,pos=(kg_width / 2 +250, kg_length/ 4 -600, kg_height-50))
    G.add_edge("weather", "weather-s", relation=relationship_describe[2], edge_class_number=4)
    G.add_node(scene[4][0], label=scene[4][0], class_number=3, pos=(kg_width / 2+250, kg_length / 4 - 700, kg_height - 150))
    G.add_edge("weather-s",scene[4][0], relation=scene[4][1], edge_class_number=4)


    #############  添加第7层 ego-information  #####
    G.add_node("ego_car", label="ego_car", class_number=2,
               pos=(kg_width / 2 - 400, kg_length/ 4*3+300, kg_height + 50))
    G.add_edge("ego_information", "ego_car",
               relation=relationship_describe[1], edge_class_number=4)


    # 遍历traffic_participants_list  和  添加ego信息   -->  添加实体
    frame_list = []
    for i, frame_i in enumerate(scene[3]):
        # 添加frame_i
        G.add_node("frame"+str(i), label="frame"+str(i), class_number=0, pos=(kg_width / 2, i*(kg_length/(scene_len-1)), kg_height-250))
        G.add_edge("scenario","frame"+str(i),relation=relationship_describe[4],edge_class_number=4)
        if i>=1:
            G.add_edge("frame" + str(i-1), "frame" + str(i), relation=relationship_describe[8], edge_class_number=0)


        #############  添加第7层 ego_information    #####
        # ego_car实体
        G.add_node("f" + str(i) + "__" + "ego", label="ego", class_number=4,
                   pos=(kg_width / 2 - 280, i*(kg_length/(scene_len-1))-350, kg_height - 500))
        G.add_edge("ego_car", "f" + str(i) + "__" + "ego", relation=relationship_describe[2], edge_class_number=4)
        G.add_edge("frame" + str(i), "f" + str(i) + "__" + "ego",relation=relationship_describe[2], edge_class_number=0)
        # 添加lane
        G.add_edge(scene[6][i][0][2], "f" + str(i) + "__" + "ego",
                   relation=relationship_describe[7], edge_class_number=4)
        # 速度等属性
        G.add_node("f" + str(i) + "__" + "ego-speed", label="speed", class_number=3,
                   pos=(kg_width / 2 - 280, i * (kg_length / (scene_len - 1)) - 350, kg_height - 600))
        G.add_edge("f" + str(i) + "__" + "ego","f" + str(i) + "__" + "ego-speed",relation=scene[6][i][0][4][0], edge_class_number=4)
        if i>=1:
            G.add_edge("f" + str(i-1) + "__" + "ego", "f" + str(i) + "__" + "ego", relation=relationship_describe[5],
                       edge_class_number=0)

        frame_i_list=[]
        for number,participant in enumerate(frame_i):
            # print(participant)
            frame_i_list.append(participant[0])
            if participant[1] == "vehicle":
                G.add_node("f" + str(i) + "__" + participant[0], label=participant[1], class_number=4,
                           pos=(kg_width / 2 +100* number, i * (kg_length / (scene_len - 1))+400, kg_height-450-50*number))
                G.add_edge("frame"+str(i), "f" + str(i) + "__" + participant[0],
                           relation=relationship_describe[2], edge_class_number=0)
            elif participant[1] == "person":
                G.add_node("f" + str(i) + "__" + participant[0], label=participant[1],
                           class_number=4,pos=(kg_width / 2+ 100 * number, i * (kg_length / (scene_len - 1))+400,
                                kg_height - 450 - 50 * number))
                G.add_edge("frame"+str(i), "f" + str(i) + "__" + participant[0],
                           relation=relationship_describe[2], edge_class_number=0)
            else:
                G.add_node("f" + str(i) + "__" + participant[0], label=participant[1],
                           class_number=4,pos=(kg_width / 2+ 100 * number, i * (kg_length / (scene_len - 1))+400,
                                kg_height -450 - 50 * number))
                G.add_edge("frame"+str(i), "f" + str(i) + "__" + participant[0],
                           relation=relationship_describe[2], edge_class_number=0)

            # 添加车道 和 类别、 自车 、 信号灯之间的联系
            G.add_edge(participant[3],"f" + str(i) + "__" + participant[0],
                       relation=relationship_describe[7], edge_class_number=4)
            G.add_edge("f" + str(i) + "__" + participant[0], participant[1],
                       relation=relationship_describe[2], edge_class_number=4)
            G.add_edge("f" + str(i) + "__" + participant[0], "f" + str(i) + "__" + "ego",
                       relation=participant[7][3], edge_class_number=0)

            # # ############################## 信号灯之间连接
            # if scene[1][3]:
            #     G.add_edge("f" + str(i) + "__" + participant[0], "light-s",
            #            relation=relationship_describe[9], edge_class_number=4)

            # 添加速度属性
            G.add_node("f" + str(i) + "__" + participant[0]+"-" +participant[4],label=participant[4], class_number=3,
                       pos=(kg_width / 2+ 70 * number, i * (kg_length / (scene_len - 1))+150,
                                kg_height - 550 - 50 * number))
            G.add_edge("f" + str(i) + "__" + participant[0],"f" + str(i) + "__" + participant[0]+"-" +participant[4],
                       relation=participant[7][0], edge_class_number=4)

            G.add_node("f" + str(i) + "__" + participant[0] + "-" + participant[5],
                       label=participant[5], class_number=3,
                       pos=(kg_width / 2 + 70 * number, i * (kg_length / (scene_len - 1)) + 350,
                            kg_height - 550 - 50 * number))
            G.add_edge("f" + str(i) + "__" + participant[0],
                       "f" + str(i) + "__" + participant[0] + "-" + participant[5],
                       relation=participant[7][2], edge_class_number=4)
            G.add_node("f" + str(i) + "__" + participant[0] + "-" + participant[6],
                       label=participant[6], class_number=3,
                       pos=(kg_width / 2 + 70 * number, i * (kg_length / (scene_len - 1)) + 550,
                            kg_height - 550 - 50 * number))
            G.add_edge("f" + str(i) + "__" + participant[0],
                       "f" + str(i) + "__" + participant[0] + "-" + participant[6],
                       relation=participant[7][1], edge_class_number=4)

            if participant[1] == "vehicle":
                G.add_node("f" + str(i) + "__" + participant[0] + "-" +  "behavior_status",
                           label= "behavior_status", class_number=3,
                           pos=(kg_width / 2 + 70 * number, i * (kg_length / (scene_len - 1)) + 750,
                                kg_height - 550 - 50 * number))
                G.add_edge("f" + str(i) + "__" + participant[0],
                           "f" + str(i) + "__" + participant[0] + "-" +  "behavior_status",
                           relation=participant[8], edge_class_number=4)


            if i >=1:
                if participant[0] in frame_list[i-1]:
                    G.add_edge("f" + str(i-1) + "__"+participant[0], "f" + str(i) + "__" + participant[0],
                           relation=relationship_describe[5], edge_class_number=0)

        frame_list.append(frame_i_list)
        # print(frame_list)

    if is_visualization:
        img = plot_AVKG_3D(G)
        print('Visualization successfully!')
    else:
        img = None

    tri = export_triple(G)
    # Optionally, print the triplets
    if is_export_triple:
        print('Now, we will print triplets!')
        for triplet in tri:
            print(triplet)
        print('Export triplets successfully!')

    if is_export_gnn:
        node_feature_matrix,edge_index_list,edge_type_list = export_node_feature_matrix(G,node_feature_number=len(node_feature))
        return G,node_feature_matrix,edge_index_list,edge_type_list,img
    else:
        return G,img
