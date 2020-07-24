# including libraries
from pyspark import SparkContext,SQLContext
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages  graphframes:graphframes:0.8.0-spark2.4-s_2.11 pyspark-shell")

from graphframes import *
from pyspark.sql import functions as f
# from pyspark.sql.functions import col, lit, when, concat, desc

# creating sql context
sc = SparkContext.getOrCreate()
sqlcontext=SQLContext(sc)
#---Importing the data sets from csv--
station_df= sqlcontext.read.format("csv").option("header", "true").csv('201508_station_data.csv')
trips_df= sqlcontext.read.format("csv").option("header", "true").csv('201508_trip_data.csv')
print("----dataset----")
station_df.show()
# creating vertices and edges
vertices = station_df.withColumnRenamed("name","id").distinct()
trip_edges=trips_df.withColumnRenamed("Start Station","src").withColumnRenamed("End Station","dst")
# creating a graph
g = GraphFrame(vertices,trip_edges)
print("----Graph----")
print(g)
# triangle count
stationTraingleCount = g.triangleCount()
print("----triangle count----")
stationTraingleCount.select("id","count").show()
# Shortest path from Stanford in Redwood City and Clay at Battery
shortPath = g.shortestPaths(landmarks=["Stanford in Redwood City","Clay at Battery"])
print("----Shortest path from Stanford in Redwood City to Clay at Battery----")
shortPath.select("id", "distances").show()
# Page rank on the datasets
stationPageRank = g.pageRank(resetProbability=0.15,tol=0.01)
print("----Page rank on the datasets----")
stationPageRank.vertices.show()
stationPageRank.edges.show()
# savig the graphs to the file
stationPageRank.vertices.coalesce(1).write.csv("vertices")
stationPageRank.edges.coalesce(1).write.csv("edges")

# Bonus
# Applying Label Propagation Algorithm on data
lpa = g.labelPropagation(maxIter=5)
print("----Applying Label Propagation Algorithm on data----")
lpa.select("id", "label").show()
# Applying BFS algorithm on the data
pathBFS = g.bfs("id = 'Japantown'","id = 'MLK Library'")
print("----Applying BFS algorithm on the data----")
pathBFS.show()
