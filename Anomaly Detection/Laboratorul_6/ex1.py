import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pyod.models.lof import LOF
data = np.loadtxt("C:/Users/Andrei/Downloads/ca-AstroPh.txt")

#1
print(data)
G = nx.Graph()
print(np.shape(data))

for points in data[:1500]:
    if G.has_edge(points[0], points[1]):
        G[points[0]][points[1]]["weight"] += 1
    else:
        G.add_edge(points[0], points[1])
        G[points[0]][points[1]]["weight"] = 1

print(G)

#2
E = []
N = []
for node in G.nodes:
    #print(node)
    ceva = nx.ego_graph(G, node)
    nx.set_node_attributes(G, {node: len(ceva.nodes)}, "N")
    nx.set_node_attributes(G, {node: len(ceva.edges)}, "E")
    nx.set_node_attributes(G, {node: ceva.size(weight="weight")}, "W")
    weights = nx.to_numpy_array(ceva, weight="weight")
    # print(weights)
    nx.set_node_attributes(G, {node: max(np.linalg.eigvals(weights).real)}, "lambda")
    #print(G.nodes[node]["lambda"])
    E.append(G.nodes[node]["E"])
    N.append(G.nodes[node]["N"])

#3
E = np.array(E)
N = np.array(N)
X = np.log(N).reshape(-1, 1)
y = np.log(E)
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
scores = (np.maximum(E, pred) / np.minimum(E, pred)) * np.log(np.abs(E - pred) + 1)
nodes_scores = {node: score for node, score in zip(G.nodes, scores)}
print(nodes_scores)

#4
sorted_scores = {node: score for node, score in sorted(nodes_scores.items(), key=lambda item: item[1], reverse=True)}
sorted_scores = dict(list(sorted_scores.items())[:10])
print(type(sorted_scores))

color_map = ["red" if node in sorted_scores else "blue" for node in G.nodes()]
plt.figure()
nx.draw(G, node_color=color_map, node_size=50, pos=None)
plt.title("Visual reprezentation Linear Regression")
plt.show()

#5
model = LOF()
features = np.array([[G.nodes[node]["E"], G.nodes[node]["N"]] for node in G.nodes])
model.fit(features)
pred = model.predict(features)
scores = np.array(scores)
norm_scores = np.linalg.norm(scores)
sum_scores = norm_scores + pred
nodes_scores = {node: score for node, score in zip(G.nodes, sum_scores)}
sorted_scores = {node: score for node, score in sorted(nodes_scores.items(), key=lambda item: item[1], reverse=True)}
sorted_scores = dict(list(sorted_scores.items())[:10])
plt.figure()
nx.draw(G, node_color=color_map, node_size=50, pos=None)
plt.title("Visual reprezentation of LOF")
plt.show()
