# Fortune's Plane Sweep Algorithm (Voronoi Diagram)

This project implements **Fortune’s Plane Sweep Algorithm** to construct the **Voronoi Diagram** in  
**O(n log n)** time using a sweep line and a dynamic beach line structure.

---

## 📌 Overview

**Fortune’s algorithm** is one of the most efficient algorithms for computing Voronoi diagrams without subdivision or brute force.

Core ideas of the algorithm:
- A horizontal **sweep line** moves from top to bottom.
- A **beach line** of parabolic arcs represents the influence of processed sites.
- Two main types of events occur:
  - **Site events** → Add a new arc.
  - **Circle events** → Remove an arc and generate a Voronoi vertex.
- The algorithm traces and finalizes **Voronoi edges** as the sweep progresses.

---

## 🚀 Features

- Full implementation of Fortune’s algorithm.
- Supports:
  - Site events  
  - Circle events  
  - Breakpoints and edge tracing  
- Optional visualization for debugging or demonstration.
- Ability to export the final diagram as images.

---

## 🧪 Example Usage

```python
from fortune import Fortune
from visualizer import Visualizer

points = [(1, 1), (9, 1), (9, 9), (1, 9)]

v = Fortune(points)
Visualizer(v).plot_sites().plot_edges().show()
