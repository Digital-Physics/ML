import random


class Centroid:
    def __init__(self, location: list[float]) -> None:
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map: dict, k: int) -> list[list[float]]:
    """ Iteratively get the k centroids of a set of data given a particular initialization.
    Bisected k-means can help avoid converging to a local, not global, min, but we don't use it here.
    Note: This function assumes our centroids will converge in 10 steps, which may not be the case.
    Note2: This function uses the Manhattan distance metric but other distance metrics could be used."""
    # for algoexpert submission
    # random.seed(42)
    initial_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    centroids = [Centroid(user_feature_map[uid]) for uid in initial_centroid_users]
    create_centroid_buckets(user_feature_map, centroids)

    for _ in range(10):
        create_centroid_buckets(user_feature_map, centroids)
        for centroid in centroids:
            center_avg = calc_cluster_avg(centroid, user_feature_map)
            centroid.location = center_avg
            centroid.closest_users.clear()

    return [centroid.location for centroid in centroids]


def create_centroid_buckets(user_feature_map: dict, centroids: list[Centroid]) -> None:
    """create clusters by finding closest Centroid for each point"""
    centroid_winner_idx = None

    # find closest centroid for each point; add points to centroid object's set
    for uid_key, features_val in user_feature_map.items():
        min_dist = float("inf")
        for i, centroid in enumerate(centroids):
            dist = manhattan_distance(centroid.location, features_val)
            if dist < min_dist:
                min_dist = dist
                centroid_winner_idx = i

        centroids[centroid_winner_idx].closest_users.add(uid_key)


def manhattan_distance(v1: list[float], v2: list[float]) -> float:
    """calc manhattan distance"""
    return sum([abs(v1[i] - v2[i]) for i in range(len(v1))])


def calc_cluster_avg(centroid: Centroid, user_feature_map: dict) -> list[float]:
    """find the cluster's average center"""
    output_vector = [0] * len(centroid.location)

    for uid in centroid.closest_users:
        for i, component in enumerate(user_feature_map[uid]):
            output_vector[i] += component

    return [component_val / len(centroid.closest_users) for component_val in output_vector]


points = {'uid_0': [-1.479359467505669, -1.895497044385029, -2.0461402601759096, -1.7109256402185178],
          'uid_1': [-1.8284426855307128, -1.714098142408679, -0.9893682669649455, -1.5766569391907947],
          'uid_2': [-1.8398933218386004, -1.7896757009107565, -1.1370177175666063, -1.0218512556903283],
          'uid_3': [-1.23224975874512, -1.8447858273094768, -1.8496517744301924, -2.4720755654344186],
          'uid_4': [-1.7714737791268318, -1.2725603446513774, -1.5512094954034525, -1.2589442628984848],
          'uid_5': [-1.359474966523909, -1.1778852627124814, -1.6106267108490018, -1.9575103480919602],
          'uid_6': [-1.4059065721891195, -1.392542330991446, -1.7702518090621264, -2.0832477098096],
          'uid_7': [-1.094110359881057, -1.5327074465621826, -1.2996267541739317, -1.5498836039723372],
          'uid_8': [-2.313871840037389, -1.5158741535880595, -1.9119086382631578, -1.8968134182320098],
          'uid_9': [-1.253421023413372, -1.8558454260261257, -1.5236216589143143, -2.2652163218494965],
          'uid_10': [-1.580484699354411, -1.7634395708076052, -1.7351800505328032, -0.8299792897991629],
          'uid_11': [-1.8230573101464664, -1.9269512846038812, -1.3174945504811744, -1.2831292800987986],
          'uid_12': [-1.306228088702581, -0.9589311907609659, -1.021715394280161, -1.8470717592593044],
          'uid_13': [-2.0278372477869957, -1.6079639813172697, -1.8008730045917802, -1.5056272040724403],
          'uid_14': [-1.5783816529934716, -1.2358509887334603, -1.2839232187299188, -1.2472946825480813],
          'uid_15': [-0.5181994365754894, -2.0162323899477474, -1.9756938738154415, -1.1142028090874305],
          'uid_16': [-1.3759800658974932, -2.0514546210658877, -1.217571230341191, -1.5464464119540413],
          'uid_17': [-1.83644794223755, -1.4819477716789196, -1.1124227782537217, -1.5285921228805897],
          'uid_18': [-1.6608692281626176, -1.920198225076026, -1.6790845011351638, -2.1978964654011537],
          'uid_19': [-2.035810312383067, -1.7148899978945003, -1.5504733439434455, -1.6550988569855354],
          'uid_20': [-1.9858504034940343, -1.0292650581191953, -1.6661578524525669, -1.258872567301933],
          'uid_21': [-1.9199529257642582, -1.5906381918172443, -1.5366574401746682, -0.6936563891868255],
          'uid_22': [-1.1551961209853039, -1.4051724105446088, -1.012830173957137, -1.1705059787675867],
          'uid_23': [-1.4486675748029776, -1.91003925191474, -0.8380433191525761, -1.175485022883859],
          'uid_24': [-1.8916228715594918, -1.2332739793185428, -1.148227807891951, -1.4011535541992968],
          'uid_25': [-0.7815638538863875, -2.2053752314854815, -2.038944512926676, -1.3554101602198685],
          'uid_26': [-2.089955450980662, -1.5822903474422771, -1.4529593555625329, -1.2189683595619265],
          'uid_27': [-1.011975686968571, -1.007355128278274, -1.192010823818011, -1.9614496834183632],
          'uid_28': [-1.503862704025446, -1.180390903332629, -1.0395108401009954, -0.6271901690634598],
          'uid_29': [-1.1096156869097429, -1.2351580015464216, -1.4591368805103075, -2.037233742083581],
          'uid_30': [-0.8824365477949329, -1.8148597267735362, -1.6790896735235126, -1.2872714884613194]}

print(get_k_means(points, 3))
