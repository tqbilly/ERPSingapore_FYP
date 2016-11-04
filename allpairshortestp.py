# the length matrix h - estimated by all-pair shortest path algorithm
INF = 999999999
RoadLen=[0.0175,0.1909,0.0667,0.1683,0.172,0.0535,0.0667,0.158,0.0667,0.1038,0.1171,0.0667,0.0915,0.1142]
shortestpath=[[ [] for i in range(9) ] for j in range(9)]
def floydWarshall(graph):
    distance = [ [ 0 for i in range(9) ] for j in range(9) ]
    for n in range(0,9):
        for k in range(0,9):
            distance[n][k] = graph[n][k]
            shortestpath[n][k].append(n)
    for k in range(0,9):
        for i in range(0,9):
            for j in range(0,9):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k]+distance[k][j]
    return distance
mygraph = [([0,INF,INF,RoadLen[0],INF,INF,INF,INF,INF]),
             ([RoadLen[1],0,INF,INF,RoadLen[2],INF,INF,INF,INF]),
             ([INF,RoadLen[3],0,INF,INF,INF,INF,INF,INF]),
             ([INF,INF,INF,0,RoadLen[4],INF,RoadLen[5],INF,INF]),
             ([INF,RoadLen[6],INF,INF,0,RoadLen[7],INF,RoadLen[8],INF]),
             ([INF,INF,RoadLen[9],INF,INF,0,INF,INF,INF]),
             ([INF,INF,INF,INF,INF,INF,0,RoadLen[10],INF]),
             ([INF,INF,INF,INF,RoadLen[11],INF,INF,0,RoadLen[12]]),
             ([INF,INF,INF,INF,INF,RoadLen[13],INF,INF,0])
             ]
h = floydWarshall(mygraph)