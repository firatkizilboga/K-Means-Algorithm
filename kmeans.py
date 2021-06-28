import numpy as np
import math

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

class K_means():
    def __init__(self):
        pass

    def fit(self, k, data):
        self.data=data
        data_copy=data.tolist().copy()
        self.clusters=[]
        
        #picks k random points from the data and puts them in self.cluster
        for i in range(k):
            choice=np.random.randint(len(data_copy))
            self.clusters.append(data_copy[choice])
            data_copy.pop(choice)
        self.clusters=np.array(self.clusters)
        del data_copy
        

        hasChanged=True
        old_clusters={}

        while hasChanged==True:
            #creates dicts for the clusters
            cluster_dict={}
            for cluster in range(len(self.clusters)):
                cluster_dict[cluster]=[]

            #calculates distance between every cluster and every data point
            for p in range(len(self.data)):
                clusters=np.array(self.clusters)
                clusters=abs(clusters-self.data[p])
                distances=[]
                for c in clusters:
                    total=0
                    for num in c:
                        total= total + num**2
                    total=math.sqrt(total)
                    distances.append(total)
                #adds the point to the closest cluster's list
                idx=distances.index(min(distances))
                cluster_dict[idx].append(p)
            
            #calculates the new positions for each cluster (centroids)
            for c in range(len(self.clusters)):
                s=np.zeros((1,self.data.shape[1]))
                for i in cluster_dict[c]:
                    s=s+(self.data[i])
                self.clusters[c]=s/len(cluster_dict[c])
            
            #checks if the centroid positions have changed
            if old_clusters==cluster_dict:
                hasChanged=False
            else:
                old_clusters=cluster_dict

    #calculates the distance and returns a list of predicted clusters
    def predict(self, data):
        predictions=[]
        
        for p in data:
            subtraction=abs(self.clusters-p)
            distances=[]
            for s in subtraction:
                total=0
                for num in s:
                    total+=num**2
                distances.append(math.sqrt(total))
            predictions.append(distances.index(min(distances)))
        return predictions


