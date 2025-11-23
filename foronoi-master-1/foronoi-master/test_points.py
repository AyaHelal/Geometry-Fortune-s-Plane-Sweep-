from foronoi import Voronoi, TreeObserver, Polygon
points = [
   (2.5, 2.5), (4, 7.5), (7.5, 2.5), (6, 7.5), (4, 4), (3, 3), (6, 3)
]
poly = Polygon(
   [(2.5, 10), (5, 10), (10, 5), (10, 2.5), (5, 0), (2.5, 0), (0, 2.5), (0, 5)]
)
v = Voronoi(poly)
# Define callback
def callback(observer, dot):
   dot.render(f"output/tree/{observer.n_messages:02d}")
# Attach observer
v.attach_observer(TreeObserver(callback=callback))
# Start diagram creation
v.create_diagram(points)