
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sympy import false


def plot_AVKG_3D(G, is_EKG=False, image_size=(800, 600)):
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
        4: (128, 0, 128),    # Purple
        8: (0, 100, 0),      # dark green
        10: (255, 0, 0)      # red
    }

    edge_color_map = {
        0: (0, 0, 255),      # Blue
        4: (100, 100, 100),  # Gray
        5: (255, 0, 0)       # Red
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
    ax.scatter(xs, ys, zs, c=node_colors, s=100, depthshade=True, edgecolors='k')

    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color=edge_colors[i], alpha=0.6)
        # Add edge labels (midpoint of the edge)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        mid_z = (pos[u][2] + pos[v][2]) / 2

        # Get the relation for the edge, assuming it's stored as 'relation' in edge attributes
        relation = G[u][v].get('relation', 'unknown')  # Default to 'unknown' if no relation attribute

        # Add text at the midpoint of the edge
        ax.text(mid_x, mid_y, mid_z, relation, fontsize=6, color='black', ha='center', va='center')

    # Add labels for nodes
    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, node, fontsize=5)

    # Plot a fixed Z plane (horizontal plane at Z_plane_height)
    x_range = np.linspace(min(xs), max(xs), 10)  # X-axis range
    y_range = np.linspace(min(ys), max(ys), 10)  # Y-axis range
    X, Y = np.meshgrid(x_range, y_range)  # Create a grid for the plane
    Z = np.full_like(X, 4800)  # Set the Z values to the fixed height
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.15)

    if is_EKG:
        Z2 = np.full_like(X, 3750)
        ax.plot_surface(X, Y, Z2, color='gray', alpha=0.15)

    ax.grid(None)
    ax.axis('on')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks
    ax.set_ylabel("Time t")

    # plt.title("Event Knowledge Graph Visualization")
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

    return img_resized  # You can return the resized image for further use


def inference_risk(EKG,is_draw_3d,kg_width=4800,kg_length=4800,kg_height=4800):
    risk_source = None       # 表示风险源是什么？主要是指向物体

    car_object = []
    car_object_index = 0
    car_object_state = []

    abnormal_car_lane = None
    abnormal_car_highrisk_lane =[]
    ego_car_lane = None
    ego_car_node = None

    abnormal_behavior_list = ["brake on the main road","change lane on the main road", "brake under the green light control" ]
    potential_risk_list = ["Ghost probe", "Cut in", "Rear-end chase"]
    focus_behavior_list = []
    has_traffic_light = False

    ###############################  find the abnormal behavior that we need to focus on  ##################################################################
    for node in EKG.neighbors("road_structure"):
        if node== "road":
            abnormal_behavior_list_index = [0,1]
            for i in abnormal_behavior_list_index:
                if abnormal_behavior_list[i] not in focus_behavior_list:
                    focus_behavior_list.append(abnormal_behavior_list[i])
            print('Now, ego car is on the main road, so focus on: ', focus_behavior_list)
    for light_node in EKG.neighbors("traffic light"):
        if light_node == "light-s" :
            edge_light = EKG.get_edge_data("light-s", "light-state").get('relation')
            if edge_light == "green":
                focus_behavior_list =[]
                abnormal_behavior_list_index = 2
                if abnormal_behavior_list[abnormal_behavior_list_index] in focus_behavior_list:
                    print("Abnormal behavior is already in the list")
                else:
                    focus_behavior_list.append(abnormal_behavior_list[abnormal_behavior_list_index])
                print('Road has traffic light, so focus on brake under the green light!')
                has_traffic_light = True
                break

    #########################  find the abnormal behavior participants from frame9 that means frame now #################################################################
    for node in EKG.nodes():
        # print(f"Node: {node}")
        if(node == "frame9"):    # 找到当前帧的节点frame
            # print(f"Node: {node}")
            for neighber1 in EKG.neighbors("frame9"):    # 找到当前节点下的实体节点
                # print(f"neighber1: {neighber1}")
                if (EKG.nodes[neighber1].get('label') != "ego"):
                    car_object.append(neighber1)
                    car_object_index +=1

                # 主要是找到关键的异常刹车行为
                for state_of_neighber in EKG.neighbors(neighber1):    # 找到实体下 的属性
                    # print(f"neighber1 --- son : {neighber1} --- {state_of_neighber}")
                    if (EKG.nodes[state_of_neighber].get('label') == "behavior_status"):  # 找到state属性
                        #  找到交通参与者的状态节点
                        # print(f"Node has behavior_status: {state_of_neighber}")
                        edge_ = EKG.get_edge_data( neighber1 , state_of_neighber).get('relation')  # 获取父节点到子节点的边数据
                        # print(f"Edge: {edge_}")

                        car_object_state.append([edge_,])

                        if (edge_ == 'hash_slow' or edge_ == 'speed_down'):
                            risk_source = neighber1
                            for pre_node in EKG.predecessors(neighber1):
                                if pre_node.split('_')[0] == "lane":
                                    abnormal_car_lane = pre_node
                                    print(f"abnormal_car_lane: {abnormal_car_lane}")
                                    abnormal_car_highrisk_lane.append(abnormal_car_lane)
                                    # 找附近的车道
                                    for lane_node in EKG.predecessors(abnormal_car_lane):
                                        if lane_node.split('_')[0] == "lane" and len(lane_node.split('_'))>1:
                                            abnormal_car_highrisk_lane.append(lane_node)
                                    for lane_node in EKG.neighbors(abnormal_car_lane):
                                        if lane_node.split('_')[0] == "lane" and len(lane_node.split('_'))>1:
                                            abnormal_car_highrisk_lane.append(lane_node)



                # 找到自车节点
                for pre_node in EKG.predecessors(neighber1):
                    # print(f"neighber1 -- pre node: {pre_node}")
                    if pre_node == "ego_car":
                        ego_car_node = neighber1


    ############################# find the normal participants lane information ##########################
    lane_list = []
    for lane_i in EKG.predecessors("lane"):
        if lane_i.split('_')[0] == "lane":
            lane_list.append(lane_i)
    print("There are lanes: ",lane_list)

    ############################# find the risk source participants historical information ##########################
    history_frame_list = ["frame0","frame1","frame2" ,"frame3","frame4","frame5","frame6","frame7","frame8","frame9"]
    risk_source_historical_lane = []
    for frame_node in history_frame_list:
        for participant in EKG.neighbors(frame_node):
            if len(participant.split('__'))>=2:
                if participant.split('__')[1] == risk_source.split('__')[1]:
                    # print(frame_node,'-->',participant)
                    for pre_participants in EKG.predecessors(participant):
                        if pre_participants.split('_')[0] == "lane":
                            # print(frame_node, '-->', participant, '-->',pre_participants)
                            if pre_participants in risk_source_historical_lane:
                                print(frame_node, '--> keeping at',pre_participants)
                            else:
                                print(frame_node, '--> change lane to', pre_participants)
                                risk_source_historical_lane.append(pre_participants)
    print('Risk source --> ',risk_source, ' has historical lane information --> ',risk_source_historical_lane)

    # 找到自车车道
    ego_car_lane_neighbors_list =[]
    for pre_node in EKG.predecessors(ego_car_node):
        # print(f"neighber1 -- pre node: {pre_node}")
        if pre_node.split('_')[0] == "lane":
            ego_car_lane = pre_node
            print(f"ego_car_lane: {ego_car_lane}")
            ego_car_lane_neighbors_list.append(ego_car_lane)
            # 找附近的车道
            for lane_node in EKG.predecessors(ego_car_lane):
                if lane_node.split('_')[0] == "lane" and len(lane_node.split('_'))>1:
                    ego_car_lane_neighbors_list.append(lane_node)
            for lane_node in EKG.neighbors(ego_car_lane):
                if lane_node.split('_')[0] == "lane" and len(lane_node.split('_'))>1:
                    ego_car_lane_neighbors_list.append(lane_node)


    other_participants_lane_list =[]
    other_participants_distance_list = []
    other_participants_angle_list = []
    other_participants_angle_list_du = []
    for participant_i in car_object:
        other_participants_angle_list.append(
            EKG.get_edge_data(participant_i, ego_car_node).get('relation'))

        for lane_of_participant_i in EKG.predecessors(participant_i):
            if lane_of_participant_i.split("_")[0] == "lane":
                other_participants_lane_list.append(lane_of_participant_i)
                # break
        for dis_of_participants_i in EKG.neighbors(participant_i):
            # print(dis_of_participants_i)
            if len(dis_of_participants_i.split("-"))>1:
                if dis_of_participants_i.split("-")[1] == "position":
                    other_participants_distance_list.append(EKG.get_edge_data(participant_i,dis_of_participants_i).get('relation'))
                    # break
                if dis_of_participants_i.split("-")[1] == "angle":
                    other_participants_angle_list_du.append(
                        EKG.get_edge_data(participant_i, dis_of_participants_i).get('relation'))

    print('Participants and their status in the frame9: ')
    print("\t",car_object)
    print("\t",car_object_state)
    print("\t",other_participants_lane_list)
    print("\t",other_participants_distance_list)
    print("\t",other_participants_angle_list)
    print("\t",other_participants_angle_list_du)
    print('Ego lane and neighbor lanes in the frame9: ')
    print("\t",ego_car_lane)
    print("\t",ego_car_lane_neighbors_list)


    lane_occ_list = risk_source_historical_lane
    for i, car_i in enumerate(car_object):
        if other_participants_angle_list_du[i] != "angle-90-180":
            if other_participants_distance_list[i] == "dis-1" or other_participants_distance_list[i] == "dis-2" or other_participants_distance_list[i] == "dis-3" or other_participants_distance_list[i] == "dis-4"\
                    or other_participants_distance_list[i] == "dis-5"or other_participants_distance_list[i] == "dis-6":
                if other_participants_lane_list[i] not in lane_occ_list:
                    lane_occ_list.append(other_participants_lane_list[i])

    for i_lane in lane_occ_list:
        lane_list.remove(i_lane)
        if i_lane in ego_car_lane_neighbors_list:
            ego_car_lane_neighbors_list.remove(i_lane)
    for i_lane in risk_source_historical_lane:
        if i_lane in lane_list:
            lane_list.remove(i_lane)
    print("Safe lane list to ego lane: ", ego_car_lane_neighbors_list)

    decision_recommendation = []
    if len(ego_car_lane_neighbors_list)>=1:
        for safe_lane in  ego_car_lane_neighbors_list:
            if safe_lane == ego_car_lane:
                decision_recommendation.append("It is recommended to slow down or brake on the"+ego_car_lane)
            else:
                decision_recommendation.append("It is recommended to change lane from "+ego_car_lane+" to "+safe_lane)
    else:
        decision_recommendation.append("It is recommended to brake on " + ego_car_lane)

######################################################################################################################
    if has_traffic_light:
        # print("Has traffic light!")
        if len(risk_source_historical_lane) == 1:
            if ego_car_lane in abnormal_car_highrisk_lane:
                print('Risk source --> ', risk_source)
                print('Potential risk --> ', potential_risk_list[0])
                print('Due to the driving environment --> Key abnormal behavior:', focus_behavior_list,
                      ' --> Find abnormal participants:',
                      risk_source, ' --> Find the historical lane information: take brake and not change the lane',
                      ' --> ', potential_risk_list[0], ' --> Find the participants lane occupancy information: ',
                      lane_occ_list
                      , ' --> Safe lane list: ', ego_car_lane_neighbors_list, ' --> Decision recommendation: ',
                      decision_recommendation)
                EKG.add_node("Risk source", label="Risk source", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2,
                                  kg_height - 1100))
                EKG.add_edge("Risk source", risk_source, relation="may_is", edge_class_number=5)

                EKG.add_node("Ghost probe", label="Ghost probe", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2 - 200,
                                  kg_height - 1200))
                EKG.add_edge("Risk source", "Ghost probe", relation="may_happen", edge_class_number=5)

                EKG.add_node("Recommendation", label="Recommendation", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2 - 400,
                                  kg_height - 1400))
                EKG.add_edge("Recommendation", "Ghost probe", relation=str(decision_recommendation),
                             edge_class_number=5)

        else:
            if ego_car_lane == risk_source_historical_lane[1]:
                print('Risk source --> ', risk_source)
                print('Potential risk --> ', potential_risk_list[1])
                if ego_car_lane == risk_source_historical_lane[1]:
                    print('Due to the driving environment --> Key abnormal behavior:', focus_behavior_list,
                          ' --> Find abnormal participants:',
                          risk_source, ' --> Find the historical lane information: change the lane',
                          ' --> ', potential_risk_list[1], ' --> Find the participants lane occupancy information: ',
                          lane_occ_list
                          , ' --> Safe lane list: ', ego_car_lane_neighbors_list, ' --> Decision recommendation: ',
                          decision_recommendation)
                    EKG.add_node("Risk source", label="Risk source", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2,
                                      kg_height - 1100))
                    EKG.add_edge("Risk source", risk_source, relation="may_is", edge_class_number=5)

                    EKG.add_node("Cut in", label="Cut in", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2 - 200,
                                      kg_height - 1200))
                    EKG.add_edge("Risk source", "Cut in", relation="may_happen", edge_class_number=5)

                    EKG.add_node("Recommendation", label="Recommendation", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2 - 400,
                                      kg_height - 1400))
                    EKG.add_edge("Recommendation", "Cut in", relation=str(decision_recommendation),
                                 edge_class_number=5)

    else:
        # print("Has not traffic light")
        if len(risk_source_historical_lane) > 1:
            print('Risk source --> ', risk_source)
            print('Potential risk --> ', potential_risk_list[1])
            if ego_car_lane == risk_source_historical_lane[1]:
                print('Due to the driving environment --> Key abnormal behavior:', focus_behavior_list, ' --> Find abnormal participants:',
                      risk_source, ' --> Find the historical lane information: change the lane',
                      ' --> ', potential_risk_list[1],' --> Find the participants lane occupancy information: ',lane_occ_list
                          ,' --> Safe lane list: ',ego_car_lane_neighbors_list,' --> Decision recommendation: ',decision_recommendation)
                EKG.add_node("Risk source", label="Risk source", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2,
                                  kg_height - 1100))
                EKG.add_edge("Risk source", risk_source, relation="may_is", edge_class_number=5)

                EKG.add_node("Cut in", label="Cut in", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2 -200,
                                  kg_height - 1200))
                EKG.add_edge("Risk source", "Cut in", relation="may_happen", edge_class_number=5)

                EKG.add_node("Recommendation", label="Recommendation", class_number=10,
                             pos=(kg_width / 2 + 100, kg_length / 2 - 400,
                                  kg_height - 1400))
                EKG.add_edge("Recommendation", "Cut in", relation=str(decision_recommendation),
                             edge_class_number=5)

        else:
            # 如果自车车道在危险的车道列表中
            if ego_car_lane in abnormal_car_highrisk_lane:
                if ego_car_lane == abnormal_car_lane:
                    print('Risk source --> ', risk_source)
                    print('Potential risk --> ', potential_risk_list[2])
                    print('Due to the driving environment --> Key abnormal behavior:', focus_behavior_list,
                          ' --> Find abnormal participants:',
                          risk_source, ' --> Find the historical lane information: take brake and not change the lane',
                          ' --> ', potential_risk_list[2],' --> Find the participants lane occupancy information: ',lane_occ_list
                          ,' --> Safe lane list: ',ego_car_lane_neighbors_list,' --> Decision recommendation: ',decision_recommendation)

                    EKG.add_node("Risk source", label="Risk source", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2,
                                      kg_height - 1100))
                    EKG.add_edge("Risk source", risk_source, relation="may_is", edge_class_number=5)

                    EKG.add_node("Rear-end chase", label="Rear-end chase", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2-200,
                                      kg_height - 1200))
                    EKG.add_edge("Risk source", "Rear-end chase", relation="may_happen", edge_class_number=5)

                    EKG.add_node("Recommendation", label="Recommendation", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2-400,
                                      kg_height - 1400))
                    EKG.add_edge("Recommendation", "Rear-end chase", relation=str(decision_recommendation), edge_class_number=5)

                else:
                    print('Risk source --> ', risk_source)
                    print('Potential risk --> ', potential_risk_list[0])
                    print('Due to the driving environment --> Key abnormal behavior:', focus_behavior_list,
                          ' --> Find abnormal participants:',
                          risk_source, ' --> Find the historical lane information: take brake and not change the lane',
                          ' --> ', potential_risk_list[0],' --> Find the participants lane occupancy information: ',lane_occ_list
                          ,' --> Safe lane list: ',ego_car_lane_neighbors_list,' --> Decision recommendation: ',decision_recommendation)
                    EKG.add_node("Risk source", label="Risk source", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2,
                                      kg_height - 1100))
                    EKG.add_edge("Risk source", risk_source, relation="may_is", edge_class_number=5)

                    EKG.add_node("Ghost probe", label="Ghost probe", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2 - 200,
                                      kg_height - 1200))
                    EKG.add_edge("Risk source", "Ghost probe", relation="may_happen", edge_class_number=5)

                    EKG.add_node("Recommendation", label="Recommendation", class_number=10,
                                 pos=(kg_width / 2 + 100, kg_length / 2 - 400,
                                      kg_height - 1400))
                    EKG.add_edge("Recommendation", "Ghost probe", relation=str(decision_recommendation),
                                 edge_class_number=5)





    # 可视化
    if is_draw_3d:
        imge = plot_AVKG_3D(EKG, is_EKG=True)

        return risk_source, imge
    else:
        return risk_source


def AVKG_3D_simple_EKG(TKG, is_draw_3d):
    if is_draw_3d:
        risk_object,imge = inference_risk(TKG,is_draw_3d)
        return risk_object,imge
    else:
        risk_object = inference_risk(TKG,is_draw_3d)
        return risk_object
