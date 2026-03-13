### KG & RGCN
A Temporal Knowledge Graphs Based RGCN Approach And Dataset for Predicting Potential Dangers in Autonomous Driving

We propose a novel, knowledge-driven framework that enables autonomous vehicles to proactively anticipate hidden risks—such as "ghost probe" incidents—by modeling driving scenarios as Temporal Knowledge Graphs (TKGs) and reasoning over them using a global feature-enhanced Relational Graph Convolutional Network (RGCN). Our method predicts potential hazards 0.4–0.8 seconds before they manifest, and provides interpretable, entity-level risk attribution and actionable decision recommendations.

## Introduction
<img src="asset/TKG_RGCN.png" width="500">
We propose an interpretable, proactive risk prediction method for autonomous driving based on Knowledge Graphs (KG) and Relational Graph Convolutional Networks (RGCN), designed to anticipate potential hazards—such as "ghost probe" scenarios—before they occur. The approach first models the dynamic driving environment as a Temporal Knowledge Graph (TKG), where nodes represent traffic participants and environmental elements, and edges encode semantic relationships along with spatio-temporal interactions. Subsequently, an enhanced RGCN model, termed TKGCN, is introduced, which integrates global scene features and employs a multi-layer, relation-aware message-passing mechanism to learn risk representations for each entity, enabling scene-level risk prediction 0.4–0.8 seconds prior to the actual hazardous event. To further improve interpretability, the system incorporates an Event Knowledge Graph (EKG) reasoning engine that identifies high-risk source entities (e.g., pedestrians suddenly stepping into the road or vehicles braking abnormally) and generates human-understandable decision rationales.

## project code
dataset/raw_image            # Display the scene image corresponding to each dataset
dataset/tkgcn_dataset        # The dataset used for training or validation

dataset_collection_tool/9_has_ghost_4lane...          # Some script code demonstrating how our dataset was collected

nets/TKGCN_V9_l  # Models 
nets/TKGCN_V9_m
nets/TKGCN_V9_s

TKG/AutoVehicle_EKG_Function_simple_version_V2           # TKG＆EKG
TKG/AutoVehicle_KG_Function_3D_simple_version_V2


train_tkgcn_transformer.py     # How we train our model


test_cutin_in_time_after_all.py    # How we test our method in a cut-in scenario
test_ghost_probe_in_time_after_all   # How we test our method in a ghost—probe scenario
