from tools.distance import EuclideanDistance, ManhattanDistance, ChebyshevDistance
from tools.voting import MajorityClassVote, InverseDistanceWeightedVote, ShepardsWorkVote
from tools.weighting import EqualWeighting, InformationGainWeighting, ReliefFWeighting
from tools.reduction import condensed_nearest_neighbor, edited_nearest_neighbor, drop2
from tools.knn import KNNClassifier