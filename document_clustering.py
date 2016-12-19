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

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering

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

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


###############################################################################
# Do the actual clustering

print("K-Means")
if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


print()
print()
print("Agglomerative Clustering Ward")
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)
ward = AgglomerativeClustering(n_clusters=true_k, linkage='ward',
                                           connectivity=connectivity)

t0 = time()
ward.fit(X.toarray())
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, ward.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, ward.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, ward.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, ward.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, ward.labels_, sample_size=1000))


print()
print()
print("Agglomerative Clustering Average Linkage")

avg = AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=true_k,
        connectivity=connectivity)
t0 = time()
avg.fit(X.toarray())
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, avg.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, avg.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, avg.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, avg.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, avg.labels_, sample_size=1000))

print()
print()
print("Spectral Clustering")

spectral = SpectralClustering(n_clusters=true_k,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")

t0 = time()
spectral.fit(X.toarray())
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, spectral.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, spectral.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, spectral.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, spectral.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, spectral.labels_, sample_size=1000))

