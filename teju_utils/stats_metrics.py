from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import scanpy as sc

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# check https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/utils/mmd.py#L25


def calculate_perc_zeros(data):
    if type(data) == str:
        data =get_array(sc.read_h5ad(data))
    # Count the total number of zeros in the expression matrix
    total_zeros = np.sum(data == 0)

    # Count the total number of elements in the expression matrix
    total_elements = data.size

    # Calculate the percentage of zeros
    percentage_zeros = (total_zeros / total_elements) * 100

    return percentage_zeros


def compute_tsne(real_cells, simulated_cells, random_seed=42):
    def metric(real_cells, simulated_cells):
        embedded_cells = TSNE().fit_transform(
                np.concatenate((real_cells, simulated_cells), axis=0)
            )

        real_embedding = embedded_cells[0 : real_cells.shape[0], :]
        fake_embedding = embedded_cells[real_cells.shape[0] :, :]
        return real_embedding,fake_embedding
    
    if type(real_cells) == str:
        real_cells =get_array(sc.read_h5ad(real_cells))
    
    if type(simulated_cells) == str:
        simulated_cells =get_array(sc.read_h5ad(simulated_cells))
    
    np.random.seed(random_seed)
    real_size=real_cells.shape[0]
    simulated_size=simulated_cells.shape[0]
    min_size=min(real_size, simulated_size)

    if real_size == simulated_size == min_size:
        real_embedding, fake_embedding = metric(real_cells, simulated_cells)

    elif real_cells.shape[0] != min_size:
        # do sampling
        for i in range(1):
            randint = np.random.randint(0, real_size, min_size)
            real_embedding, fake_embedding = metric(real_cells[randint], simulated_cells)
    else:
        for i in range(1):
            randint = np.random.randint(0, simulated_size, min_size)
            real_embedding, fake_embedding = metric(real_cells, simulated_cells[randint])
    
    return real_embedding, fake_embedding




def compute_pairwise_distance(real_cells, simulated_cells, random_seed=42, bootstrap=5):

    def metric(real_cells, simulated_cells):
        # minmax = MinMaxScaler()
        # minmax.fit(np.concatenate([real_cells, simulated_cells]))
        
        # real_cells = minmax.transform(real_cells)
        # simulated_cells = minmax.transform(simulated_cells)

        dist = DistanceMetric.get_metric('euclidean')
        d = dist.pairwise(simulated_cells, real_cells)
        return d.mean()
    

    if type(real_cells) == str:
        real_cells =get_array(sc.read_h5ad(real_cells))
    
    if type(simulated_cells) == str:
        simulated_cells =get_array(sc.read_h5ad(simulated_cells))

    score = do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric)
    return score

def do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric):
    if type(real_cells) == str:
        real_cells =get_array(sc.read_h5ad(real_cells))
    
    if type(simulated_cells) == str:
        simulated_cells =get_array(sc.read_h5ad(simulated_cells))

    np.random.seed(random_seed)
    real_size=real_cells.shape[0]
    simulated_size=simulated_cells.shape[0]
    min_size=min(real_size, simulated_size)
    
    if real_size == simulated_size == min_size:
        score = metric(real_cells, simulated_cells)

    elif real_cells.shape[0] != min_size:
        # do sampling
        scores=[]
        
        for i in range(bootstrap):
            randint = np.random.randint(0, real_size, min_size)
            score = metric(real_cells[randint], simulated_cells)
            scores.append(score)
        
        score = sum(scores)/len(scores)

    else:
        scores=[]
        for i in range(bootstrap):
            randint = np.random.randint(0, simulated_size, min_size)
            score = metric(real_cells, simulated_cells[randint])
            scores.append(score)
        
        score = sum(scores)/len(scores)
    return score

def compute_knn_distance(real_cells, simulated_cells, k=1,  random_seed=42, bootstrap=5):
    def metric(real_cells, simulated_cells):
        minmax = MinMaxScaler()
        minmax.fit(np.concatenate([real_cells, simulated_cells]))
        
        real_cells = minmax.transform(real_cells)
        simulated_cells = minmax.transform(simulated_cells)

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(real_cells)

        d_knn, _ = nn.kneighbors(simulated_cells)
        # print(f"{k}-NN distance: {d_knn.mean()}")
        return d_knn.mean()

    score = do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric)
    
    return score



def compute_cosine(real_cells, simulated_cells, random_seed=42, bootstrap=5):
    def metric(real_cells, simulated_cells):
        centroid_real = np.mean(real_cells, axis=0)
        centroid_simulated = np.mean(simulated_cells, axis=0)

        cosine_dist = cosine(centroid_real, centroid_simulated)

        # print(f'Cosine Distance: {cosine_dist}')
        return cosine_dist
    score = do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric)
    
    return score

def compute_euclidean(real_cells, simulated_cells, random_seed=42, bootstrap=5):
    def metric(real_cells, simulated_cells):
        centroid_real = np.mean(real_cells, axis=0)
        centroid_simulated = np.mean(simulated_cells, axis=0)
        
        euclidean_dist = euclidean(centroid_real, centroid_simulated)

        # print(f'Euclidean Distance: {euclidean_dist}')
        return euclidean_dist
    
    score = do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric)
    
    return score

def compute_median_distance(Y, k=25):
    # Compute the average distance between each point and its nearest k neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
    distances, _ = nbrs.kneighbors(Y)
    # avg_distances = np.mean(distances, axis=1)

    # return np.median(avg_distances)
    return np.median(distances[:, k-1])

def gaussian_kernel(x, y, sigma):

    # return  np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma**2))
    return  np.exp(-cdist(x, y, 'sqeuclidean') / (sigma**2))


def sum_of_gaussian_kernels(X, Y, sigmas):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for sigma in sigmas:
        K += gaussian_kernel(X, Y, sigma)
    return K

def compute_mmd(X, Y, random_seed=42, bootstrap=5):
    def metric(X, Y):
        m = compute_median_distance(X)
        sigmas = [m / 0.5, m, m/2]
        
        Kxx = sum_of_gaussian_kernels(X, X, sigmas)
        Kyy = sum_of_gaussian_kernels(Y, Y, sigmas)
        Kxy = sum_of_gaussian_kernels(X, Y, sigmas)
        
        mmd = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
        return mmd
    
    score = do_bootstrap(X, Y, random_seed, bootstrap, metric)
    
    return score


def compute_auroc_rf(real_cells, simulated_cells, random_seed=42, bootstrap=5):
    def metric(real_cells, simulated_cells):
        real_labels = np.ones(real_cells.shape[0])
        simulated_labels = np.zeros(simulated_cells.shape[0])

        data = np.concatenate((real_cells, simulated_cells), axis=0)
        labels = np.concatenate((real_labels, simulated_labels), axis=0)

        pca = PCA(n_components=50, random_state=random_seed)
        data_pca = pca.fit_transform(data)

        split_index = int(0.8 * len(data_pca))
        randint = np.random.randint(0, len(data_pca), split_index)
        X_train, X_test, y_train, y_test = train_test_split(data_pca, labels, test_size=0.2, random_state=random_seed)

        rf = RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=random_seed)
        rf.fit(X_train, y_train)

        y_pred_prob = rf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_prob)
        return auroc

    # Bootstrap the process for stability
    score = do_bootstrap(real_cells, simulated_cells, random_seed, bootstrap, metric)
    return score


def get_array(data):
    from scipy.sparse import issparse
    if issparse(data):
        return data.X.toarray()
    else:
        return data.X
    
# def gower_similarity(pt1, pt2, ranges):
#     # https://github.com/a-skabar/TDS-EvalSynthData/blob/main/master.py
#     # Returns Gower similarity between two points
#     # Parameters:
#     # pt1 (1d numpy array): First datapoint
#     # pt2 (1d numpy array): Second datapoint
#     # ranges (1d numpy array): The ranges for numeric variables
#     # Returns:
#     # sim (float): Gower similarity
#     n_dims = pt1.size
#     psi_sum =  0.0
#     for i in range(n_dims):
#         if ranges[i] != None:
#             # if ranges[i] == 0:
#             #     psi = np.array(1.0 - 0.0)
#             # else:
#             psi = np.array(1.0 - (np.abs(float(pt1[i]) - float(pt2[i])) / ranges[i]))
#         else:
#             psi = 1.0 if pt1[i] == pt2[i] else 0.0
    
#         psi_sum += psi
#     sim = psi_sum / n_dims
#     return sim


# def average_max_similarity_internal(data, ranges):
#     n_points = data.shape[0]
#     max_sims = []
#     for i in range(n_points):
#         sims = []
#         for j in range(n_points):
#             if i != j:
#                 sims.append(gower_similarity(data[i, :], data[j, :], ranges))
#         max_sims.append(np.max(sims))
#     return max_sims
     
# def average_max_similarity_cross(samples, data, ranges):
#     n_samples = samples.shape[0]
#     n_data = data.shape[0]
#     max_sims = []
#     for i in range(n_samples):
#         sims=[]
#         for j in range(n_data):
#             sims.append(gower_similarity(samples[i, :], data[j, :], ranges))
#         max_sims.append(np.max(sims))
#     return max_sims       

# def gower_score(real_cells, synthetic_cells):
#     tmp = np.vstack([real_cells, synthetic_cells])
#     ndims = tmp.shape[1]
#     ranges = np.array([(np.max(tmp[:, x]) - np.min(tmp[:, x])) if np.isreal(tmp[0,x]) else None for x in range(ndims)])
#     sim_synth = average_max_similarity_internal(synthetic_cells, ranges)
#     sim_real = average_max_similarity_internal(real_cells, ranges)
#     sim_cross = average_max_similarity_cross(synthetic_cells, real_cells, ranges)
#     g = np.mean(sim_cross)/np.mean(sim_real)
#     return g, sim_real, sim_synth, sim_cross 


