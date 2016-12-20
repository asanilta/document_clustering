from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, AffinityPropagation, Birch

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import os


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Create Dataset and Extract Features

labels = []
target_names = []
data = []
root = os.path.dirname(os.path.realpath(__file__))+"/Dataset"

i = 0
for dirname in os.listdir(root):
    target_names.append(dirname)
    for file in os.listdir(os.path.join(root,dirname)):
        data.append(open(root+"/"+dirname+"/"+file).read())
        labels.append(i)
    i+=1

labels = np.array(labels)
target_names = np.array(target_names)

print("%d documents" % len(data))
print("%d categories" % len(target_names))
print()

true_k = np.unique(labels).shape[0]

t0 = time()
tfidfVectorizer = TfidfVectorizer(max_df=0.8, max_features=1000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
tfidf = tfidfVectorizer.fit_transform(data)

print("TF*IDF")
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % tfidf.shape)
print()
print()

t0 = time()
tfVectorizer = TfidfVectorizer(max_df=0.8, max_features=1000,
                                 min_df=2, stop_words='english',
                                 use_idf=False)
tf = tfVectorizer.fit_transform(data)

print("TF")
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % tf.shape)
print()
print()

t0 = time()
binarytfVectorizer = TfidfVectorizer(max_df=0.8, max_features=1000,
                                 min_df=2, stop_words='english',
                                 use_idf=True,binary=True)
binarytf = binarytfVectorizer.fit_transform(data)

print("(Binary TF)*IDF")
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % binarytf.shape)
print()
print()

binaryVectorizer = TfidfVectorizer(max_df=0.8, max_features=1000,
                                 min_df=2, stop_words='english',
                                 use_idf=True,binary=True)
binary = binaryVectorizer.fit_transform(data)

print("Binary")
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % binary.shape)
print()
print()

feature_names = ["TF*IDF","TF", "(Binary TF)*IDF", "Binary"]
features = [tfidf,tf,binarytf,binary]
vectorizers = [tfidfVectorizer,tfVectorizer,binarytfVectorizer,binaryVectorizer]

# if opts.n_components:
#     print("Performing dimensionality reduction using LSA")
#     t0 = time()
#     # Vectorizer results are normalized, which makes KMeans behave as
#     # spherical k-means for better results. Since LSA/SVD results are
#     # not normalized, we have to redo the normalization.
#     svd = TruncatedSVD(opts.n_components)
#     normalizer = Normalizer(copy=False)
#     lsa = make_pipeline(svd, normalizer)

#     X = lsa.fit_transform(X)

#     print("done in %fs" % (time() - t0))

#     explained_variance = svd.explained_variance_ratio_.sum()
#     print("Explained variance of the SVD step: {}%".format(
#         int(explained_variance * 100)))

#     print()


###############################################################################
# Clustering Algorithms

minikm = MiniBatchKMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10,
                             init_size=1000, batch_size=1000, verbose=opts.verbose, random_state=0)
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10,
                verbose=opts.verbose, random_state=0)
spectral = SpectralClustering(n_clusters=true_k,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
dbscan = DBSCAN(eps=.35)
birch = Birch(n_clusters=true_k)  

algorithms = [minikm,km,spectral,dbscan,birch]
algo_names = ["Mini Batch K-Means","K-Means", "Spectral", "DBSCAN", "Birch"]


for algo_name, algo in zip(algo_names,algorithms):
  for feature_name, feature in zip(feature_names, features):
    print("Clustering Algorithm: " + algo_name)
    print("Feature: " + feature_name)    
    print("Clustering data with %s" % algo)
    t0 = time()
    algo.fit(feature)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, algo.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, algo.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, algo.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, algo.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(feature, algo.labels_, sample_size=1000))
    print()
    print()

for feature_name, feature in zip(feature_names, features):
  # connectivity matrix for structured Ward
  connectivity = kneighbors_graph(feature, n_neighbors=10, include_self=False)
  # make connectivity symmetric
  connectivity = 0.5 * (connectivity + connectivity.T)
  algo = AgglomerativeClustering(n_clusters=true_k, linkage='ward',connectivity=connectivity)
  print("Clustering Algorithm: Agglomerative (Ward)")
  print("Feature: " + feature_name)    
  print("Clustering data with %s" % algo)
  t0 = time()
  algo.fit(feature.toarray())
  print("done in %0.3fs" % (time() - t0))
  print()
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, algo.labels_))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, algo.labels_))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, algo.labels_))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, algo.labels_))
  print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(feature, algo.labels_, sample_size=1000))
  print()
  print()

  # print("Top terms per cluster:")

  # if opts.n_components:
  #     original_space_centroids = svd.inverse_transform(km.cluster_centers_)
  #     order_centroids = original_space_centroids.argsort()[:, ::-1]
  # else:
  #     order_centroids = km.cluster_centers_.argsort()[:, ::-1]

  # terms = vectorizer.get_feature_names()
  # for i in range(true_k):
  #     print("Cluster %d:" % i, end='')
  #     for ind in order_centroids[i, :10]:
  #         print(' %s' % terms[ind], end='')
  #     print()
  # print()
  # print()