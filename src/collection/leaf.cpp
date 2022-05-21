#include <odesolver/collection/leaf.hpp>


namespace odesolver {
    namespace collections {
        Leaf::Leaf(const std::vector<int> indices) :
            indices_(indices), depth_(indices.size() - 1)
        {}

        void Leaf::info() const {
            std::cout << "\nSolution: " << std::endl;
            std::cout << "\tDepth: " << depth_ << std::endl;
            std::cout << "\tIndices:";
            for(const auto& index: indices_)
                std::cout << " " << index;
            std::cout << std::endl;
        }

        int Leaf::get_ith_depth_index(int i) const
        {
            return indices_[i];
        }

        json Leaf::to_json() const
        {
            return json {{"depth", depth_},
                         {"indices", indices_}};
        }

        void write_leaves_to_file(std::string rel_dir, std::vector<Leaf*> leaves)
        {
            json j;
            for(auto &sol: leaves)
                j.push_back(sol->to_json());
            param_helper::fs::write_parameter_file(json {{"number_of_leaves", leaves.size()}, {"leaves", j}}, param_helper::proj::project_root() + rel_dir + "/", "leaves", false);
        }

        std::vector<std::shared_ptr<Leaf>> load_leaves(std::string rel_dir)
        {
            std::vector<std::shared_ptr<Leaf>> leaves;
            
            json j = param_helper::fs::read_parameter_file(param_helper::proj::project_root() + rel_dir + "/", "leaves", false);
            leaves.reserve(j["number_of_leaves"].get<int>());
            for(auto &sol: j["leaves"])
                leaves.push_back(std::make_shared<Leaf>(sol["indices"].get<std::vector<int>>()));
            std::cout << "leaves loaded" << std::endl;
            return leaves;
        }
    }
}
