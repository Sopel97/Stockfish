#include <fstream>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <iostream>

struct Node {
  // Inputs are in an array.
  uint32_t input_id;

  // children[0] = right, children[1] = left.
  // chosen by children[v <= cutoff]
  uint32_t children[2];

  int32_t cutoff;

  // The value that is accumulated over all trees in a forest.
  float value;

  // A sentinel that loops to itself and doesn't add anything
  Node(uint32_t node_id) :
    input_id(0),
    children{node_id, node_id},
    cutoff(std::numeric_limits<int32_t>::max()),
    value(0.0f)
  {

  }

  Node(uint32_t /* tree_id */, uint32_t node_id, std::istream& is) {
    bool nan_go_left;
    bool is_leaf;
    float cutoff_float;

    is >> value >> input_id >> cutoff_float >> nan_go_left >> children[1] >> children[0] >> is_leaf;

    cutoff = static_cast<int32_t>(std::floor(cutoff_float));

    if (is_leaf) {
      children[0] = node_id;
      children[1] = node_id;
      cutoff = std::numeric_limits<int32_t>::max();
      input_id = 0;
    } else {
      value = 0.0f;
    }
  }
};

static_assert(sizeof(Node) == 20);

struct Forest {
  // Nodes are stored contiguously. It allows easier vectorization.
  // The `trees_starts` holds indices of the first nodes in each tree.
  // We "join" the trees such that a bunch of them can be traversed
  // by one loop. Also we split them into 4 "supertrees" (which are not really trees)
  // so that we can reduce the amount of instruction dependencies in the loop.
  // The vectorization can be performed in this way too, but it requires
  // first moving all the nodes into one vector.
  std::vector<Node> nodes[4];
  std::vector<uint32_t> trees_starts[4];
  float bias;

  Forest(uint32_t /* forest_id */, float bias_, std::istream& is) :
    bias(bias_)
  {
    uint32_t num_trees;
    is >> num_trees;

    for (uint32_t tree_id = 0; tree_id < num_trees; ++tree_id) {
      uint32_t bucket = tree_id % 4;
      uint32_t num_nodes;
      is >> num_nodes;

      const uint32_t tree_root_node = nodes[bucket].size();

      if (!trees_starts[bucket].empty()) {
        const uint32_t prev_tree_root_node = trees_starts[bucket].back();
        for (uint32_t node_id = prev_tree_root_node; node_id < tree_root_node; ++node_id) {
          // A hacky condition for is_leaf
          if (nodes[bucket][node_id].children[0] == nodes[bucket][node_id].children[1]) {
            // Attach the previous tree leaves to the next tree root.
            nodes[bucket][node_id].children[0] = tree_root_node;
            nodes[bucket][node_id].children[1] = tree_root_node;
          }
        }
      }

      trees_starts[bucket].push_back(tree_root_node);

      for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
        auto& node = nodes[bucket].emplace_back(tree_id, node_id, is);
        // We're making one big graph, so adjust the children indices
        node.children[0] += tree_root_node;
        node.children[1] += tree_root_node;
      }
    }

    // Insert sentinel nodes at the end so that if this bucket is fully processed
    // and the other buckets are being processed we don't add node.value repeatedly
    for (uint32_t bucket = 0; bucket < 4; ++bucket) {
      const uint32_t sentinel_node_id = nodes[bucket].size();

      if (!trees_starts[bucket].empty()) {
        const uint32_t prev_tree_root_node = trees_starts[bucket].back();
        for (uint32_t node_id = prev_tree_root_node; node_id < sentinel_node_id; ++node_id) {
          // A hacky condition for is_leaf
          if (nodes[bucket][node_id].children[0] == nodes[bucket][node_id].children[1]) {
            // Attach the previous tree leaves to the sentinel.
            nodes[bucket][node_id].children[0] = sentinel_node_id;
            nodes[bucket][node_id].children[1] = sentinel_node_id;
          }
        }
      }

      nodes[bucket].emplace_back(sentinel_node_id);
    }
  }

  [[nodiscard]] float evaluate(int32_t input[]) const {
    float v0 = bias;
    float v1 = 0.0f;
    float v2 = 0.0f;
    float v3 = 0.0f;
    uint32_t node_id0 = 0;
    uint32_t node_id1 = 0;
    uint32_t node_id2 = 0;
    uint32_t node_id3 = 0;
    for (;;) {
      const auto& node0 = nodes[0][node_id0];
      const auto& node1 = nodes[1][node_id1];
      const auto& node2 = nodes[2][node_id2];
      const auto& node3 = nodes[3][node_id3];

      const int32_t in0 = input[node0.input_id];
      const int32_t in1 = input[node1.input_id];
      const int32_t in2 = input[node2.input_id];
      const int32_t in3 = input[node3.input_id];

      const uint32_t choice0 = in0 <= node0.cutoff;
      const uint32_t choice1 = in1 <= node1.cutoff;
      const uint32_t choice2 = in2 <= node2.cutoff;
      const uint32_t choice3 = in3 <= node3.cutoff;

      const uint32_t next_node_id0 = node0.children[choice0];
      const uint32_t next_node_id1 = node1.children[choice1];
      const uint32_t next_node_id2 = node2.children[choice2];
      const uint32_t next_node_id3 = node3.children[choice3];

      v0 += node0.value;
      v1 += node1.value;
      v2 += node2.value;
      v3 += node3.value;

      if (
           node_id0 == next_node_id0
        && node_id1 == next_node_id1
        && node_id2 == next_node_id2
        && node_id3 == next_node_id3
        ) {
        break;
      }

      node_id0 = next_node_id0;
      node_id1 = next_node_id1;
      node_id2 = next_node_id2;
      node_id3 = next_node_id3;
    }
    return v0 + v1 + v2 + v3;
  }
};

struct Classifier {
  std::vector<Forest> forests;

  Classifier(std::istream& is) {
    uint32_t num_classes;
    is >> num_classes;
    std::vector<float> biases(num_classes);
    // in the future spec this should be a part of each forest, not all at the start
    for (uint32_t i = 0; i < num_classes; ++i)
      is >> biases[i];

    for (uint32_t i = 0; i < num_classes; ++i)
      forests.emplace_back(i, biases[i], is);
  }

  [[nodiscard]] float evaluate_class_probability(int32_t input[], uint32_t class_id) {
    if (forests.size() == 1) {
      switch (class_id) {
        case 0:
          return make_prob(forests[0].evaluate(input));
        case 1:
          return 1.0f - make_prob(forests[0].evaluate(input));
        default:
          return 0.0f;
      }
    } else {
      if (class_id >= forests.size()) {
        return 0.0f;
      }

      std::vector<float> evals;
      evals.reserve(forests.size());
      float sum = 0.0f;

      for (auto& forest : forests) {
        const float eval = forest.evaluate(input);
        evals.push_back(eval);
        sum += std::exp(eval);
      }

      return std::exp(evals[class_id] - std::log(sum));
    }
  }

private:
  [[nodiscard]] float make_prob(float x) { return 1.0f / (1.0f + std::exp(x)); }
};

[[nodiscard]] inline float step13_prunable(int32_t input[]) {
  static Classifier model = [](){
    std::ifstream is("sample.txt");
    return Classifier(is);
  }();
  return model.evaluate_class_probability(input, 1);
}
