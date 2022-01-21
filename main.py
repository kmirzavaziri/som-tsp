from random import randrange, shuffle
import matplotlib.pyplot as plot
import itertools

FILE = '1'
COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]


def euclidean_distance(a, b):
    return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** .5


def cmp(n):
    ''' Circular Mirrorable Permutations of numbers 1 to n '''
    if n == 3:
        return [[1, 2, 3]]

    ps = cmp(n - 1)
    r = []
    for p in ps:
        for i in range(n - 1):
            t = p.copy()
            t.insert(i, n)
            r.append(t)
    return r


def cm_permutations(l):
    n = len(l)
    return [[l[p[i] - 1] for i in range(n)] for p in cmp(n)]


class Visualizer:
    def __init__(self, rows, cols):
        self.fig, self.axs = plot.subplots(rows, cols)
        self.axs = self.axs.flatten()
        self.counter = 0

        self.fig.set_size_inches(20, 10)
        for ax in self.axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        plot.subplots_adjust(wspace=0, hspace=0.3)

    def add(self, cities, *, roads=[], cluster_centers=[], title=''):
        '''Visualize the given cities colored by their clusters'''
        self.axs[self.counter].set_title(title)
        self.axs[self.counter].scatter(
            *zip(*(list(map(lambda x: x['position'], cities)) + cluster_centers)),
            facecolors=list(map(lambda x: COLORS[x['cluster']] if x['cluster'] is not None else 'black', cities)) +
            ['none'] * len(cluster_centers),
            edgecolors=['none'] * len(cities) +
            [COLORS[cluster_centers.index(cluster_center)] for cluster_center in cluster_centers]
        )
        for road in roads:
            self.axs[self.counter].plot(
                [road[0]['position'][0], road[1]['position'][0]],
                [road[0]['position'][1], road[1]['position'][1]],
                color=COLORS[road[0]['cluster']]
                if road[0]['cluster'] == road[1]['cluster'] and road[0]['cluster'] is not None
                else 'black'
            )

        extent = self.axs[self.counter].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(f'{FILE}/{self.counter}.png', bbox_inches=extent.expanded(1, 1.3), dpi=300)

        self.counter += 1

    def show(self):
        self.fig.savefig(f'{FILE}/all.png', dpi=300)
        plot.show()


class SOM:
    def __init__(self, random_range, input_size, output_size, alpha, nu):
        '''initiate the weights'''
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.nu = nu
        self.w = [[randrange(*random_range[i]) for i in range(input_size)] for j in range(output_size)]

    def compute(self, input):
        '''Compute the output of the SOM based on the input'''
        return [euclidean_distance(input, self.w[i]) for i in range(self.output_size)]

    def self_optimize(self, input, min_output_index):
        '''Learn From the given input and the minimum output (unsupervised)'''
        for i in range(self.output_size):
            if euclidean_distance(self.w[min_output_index], self.w[i]) <= self.nu:
                for j in range(self.input_size):
                    self.w[i][j] += self.alpha * (input[j] - self.w[i][j])


# Initiate the visualizer
visualizer = Visualizer(3, 4)

# Read and visualize the problem
with open(FILE + '.tsp') as f:
    cities = [{'position': tuple(map(float, l.split()[1:3])), 'cluster': None} for l in f.readlines()]
visualizer.add(cities, title='Cities')

# Initiate the SOM
DIMENSION = 2
CLUSTERS_COUNT = 5
ALPHA = .4
NU = 9
RANDOM_RANGE = [
    [min(map(lambda x: x['position'][i], cities)), max(map(lambda x: x['position'][i], cities)) + 1]
    for i in range(DIMENSION)
]
som = SOM(RANDOM_RANGE, DIMENSION, CLUSTERS_COUNT, ALPHA, NU)

# Iterate and cluster
iteration = 0
while True:
    iteration += 1

    # Cluster each city
    for city in cities:
        som_output = som.compute(city['position'])
        city['cluster'] = som_output.index(min(som_output))

    # Visualize the clustered cities
    visualizer.add(cities, cluster_centers=som.w, title=f'Iteration {iteration}')

    # Learn from each city
    for city in cities:
        som.self_optimize(city['position'], city['cluster'])

    # Lower the degree of learning (alpha)
    # and the threshold of neighborhood (nu)
    som.alpha -= .01
    som.nu -= 1

    # Exit Condition
    if som.alpha <= 0 or som.nu <= 0:
        break

# Organize cities based on clusters
clusters = list(filter(lambda x: x, [
    {'cities': [city for city in cities if city['cluster'] == i]}
    for i in range(CLUSTERS_COUNT)
]))

# Find the optimal path in each cluster
cost = 0
roads = []
for cluster in clusters:
    # If the cluster is small and it is possible, use brute-force and check all permutations and find the best.
    if len(cluster) < 7:
        permutations = list(map(list, itertools.permutations(cluster['cities'])))
        costs = [
            sum([euclidean_distance(p[j]['position'], p[j + 1]['position']) for j in range(len(p) - 1)])
            for p in permutations
        ]
        best_cost = min(costs)
        best_permutation = permutations[costs.index(best_cost)]
    # Else, just use a random permutation. (This part can be better if we use other optimization algorithms as a mix.)
    else:
        best_permutation = cluster['cities'].copy()
        shuffle(best_permutation)
        best_cost = sum([
            euclidean_distance(best_permutation[j]['position'], best_permutation[j + 1]['position'])
            for j in range(len(best_permutation) - 1)
        ])

    # Add this path to roads array
    roads += [(best_permutation[j], best_permutation[j + 1]) for j in range(len(best_permutation) - 1)]

    # Add two endpoints of the path to cluster data for connecting clusters
    cluster['ends'] = (best_permutation[0], best_permutation[-1])

    # Add the cost of this part
    cost += best_cost

visualizer.add(cities, roads=roads, title=f'Intracluster Roads')

# Connect clusters (Using brute-force to find the best possible way)
options = [list(map(int, format(i, f'0{CLUSTERS_COUNT}b'))) for i in range(2 ** CLUSTERS_COUNT)]
permutations = cm_permutations(clusters)
intercluster_cost = None
intercluster_roads = []
for p in permutations:
    for option in options:
        po_cost = 0
        po_roads = []
        for j in range(len(p)):
            po_cost += euclidean_distance(p[j]['ends'][1 - option[j]]['position'],
                                          p[j - 1]['ends'][option[j - 1]]['position'])
            po_roads += [(p[j]['ends'][1 - option[j]], p[j - 1]['ends'][option[j - 1]])]
        if intercluster_cost is None or po_cost < intercluster_cost:
            intercluster_cost = po_cost
            intercluster_roads = po_roads

roads += intercluster_roads
cost += intercluster_cost

visualizer.add(cities, roads=roads, title=f'All Roads')
visualizer.fig.suptitle(f'Cost = {cost}')

visualizer.show()
