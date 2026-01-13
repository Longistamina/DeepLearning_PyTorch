'''
Watch this video:
+ How to use edge features in GNN: https://www.youtube.com/watch?v=mdWQYYapvR8&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=5

########################

Example of edge features

Social Network Graph with Edge Features

Nodes (Users):
- Node 0: Alice
- Node 1: Bob
- Node 2: Charlie
- Node 3: Diana

Node Features (3 features per node):
Node 0: [25, 1, 100]  # [age, gender(1=female), num_posts]
Node 1: [30, 0, 250]  # [age, gender(0=male), num_posts]
Node 2: [28, 0, 180]  # [age, gender(0=male), num_posts]
Node 3: [22, 1, 320]  # [age, gender(1=female), num_posts]

Edges (Friendships):
Edge 0→1: Alice to Bob
Edge 0→2: Alice to Charlie
Edge 1→3: Bob to Diana
Edge 2→3: Charlie to Diana

    (0)────────────(1)
     │              │
     │              │
     └────(2)───────┘
           │
          (3)

Edge Features (2 features per edge):
Edge 0→1: [5.0, 12]  # [years_friends, messages_per_week]
Edge 0→2: [3.5, 8]   # [years_friends, messages_per_week]
Edge 1→3: [2.0, 25]  # [years_friends, messages_per_week]
Edge 2→3: [1.5, 15]  # [years_friends, messages_per_week]

#------
## In matrix form:
#------

Node feature matrix X (4 nodes × 3 features):
[[25,  1, 100],
 [30,  0, 250],
 [28,  0, 180],
 [22,  1, 320]]

Edge index:
[[0, 0, 1, 2],   # source node
 [1, 2, 3, 3]]   # target node

Edge feature matrix E (4 edges × 2 features):
[[5.0, 12],
 [3.5,  8],
 [2.0, 25],
 [1.5, 15]]

During GNN message passing, when Node 1 (Bob) aggregates messages:
- From Node 0 (Alice): uses edge features [5.0, 12] along with Alice's node features
- These edge features can weight or transform the message differently than edges with different characteristics
'''