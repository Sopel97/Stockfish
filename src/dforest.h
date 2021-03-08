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
  // The `treess_starts` holds indices of the first nodes in each tree.
  std::vector<Node> nodes;
  std::vector<uint32_t> trees_starts;
  float bias;

  Forest(uint32_t /* forest_id */, float bias_, std::istream& is) :
    bias(bias_)
  {
    uint32_t num_trees;
    is >> num_trees;

    for (uint32_t tree_id = 0; tree_id < num_trees; ++tree_id) {
      uint32_t num_nodes;
      is >> num_nodes;

      const uint32_t tree_root_node = nodes.size();

      if (!trees_starts.empty()) {
        const uint32_t prev_tree_root_node = trees_starts.back();
        for (uint32_t node_id = prev_tree_root_node; node_id < tree_root_node; ++node_id) {
          // A hacky condition for is_leaf
          if (nodes[node_id].children[0] == nodes[node_id].children[1]) {
            // Attach the previous tree leaves to the next tree root.
            nodes[node_id].children[0] = tree_root_node;
            nodes[node_id].children[1] = tree_root_node;
          }
        }
      }

      trees_starts.push_back(tree_root_node);

      for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
        auto& node = nodes.emplace_back(tree_id, node_id, is);
        // We're making one big graph, so adjust the children indices
        node.children[0] += tree_root_node;
        node.children[1] += tree_root_node;
      }
    }
  }

  [[nodiscard]] float evaluate(int32_t input[]) const {
    float v = bias;
    uint32_t node_id = 0;
    for (;;) {
      const auto& node = nodes[node_id];
      const int32_t in = input[node.input_id];
      const uint32_t choice = in <= node.cutoff;
      const uint32_t next_node_id = node.children[choice];
      v += node.value;

      if (node_id == next_node_id)
        break;

      node_id = next_node_id;
    }
    return v;
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
