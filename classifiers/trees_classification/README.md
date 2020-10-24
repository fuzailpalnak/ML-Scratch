# Tree Classification

### KD Nearest Neighbour Classification

#### Tree Generation

- Split data along the axis, most commonly median is used as a measure to split
- Continue Splitting until reached last node

#### Neighbour Query

- a = Compute the square distance between current point and point for which neighbour is to be found
- Update the nearest neighbour if a < found neighbour
- Based on threshold, decide which side of the tree to traverse
- If Distance to root node is < distance to neighbour, traverse on the other section of tree