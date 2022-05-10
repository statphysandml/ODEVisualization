#ifndef PROJECT_COLLECTIONCOUNTER_HPP
#define PROJECT_COLLECTIONCOUNTER_HPP

#include <vector>
#include <iostream>

// https://stackoverflow.com/questions/1926605/how-to-count-the-number-of-objects-created-in-c


namespace odesolver {
    namespace collections {
        /** @brief Class allowing for an inspection of the number of generated and deleted collections. */
        template <typename T>
        struct Counter
        {
            Counter(const int depth)
            {
                objects_created_[depth]++;
                objects_alive_[depth]++;
            }

            static std::vector<int> objects_created_;
            static std::vector<int> objects_alive_;

            static void print_statistics()
            {
                number_of_created_collections_per_depth();
                number_of_alive_collections_per_depth();
            }

            static void number_of_created_collections_per_depth()
            {
                std::cout << "Number of created collections per depth:" << std::endl;
                for(auto i = 0; i < objects_created_.size(); i++)
                {
                    if(objects_created_[i] > 0)
                        std::cout << " " << "Depth " << i << ": " << objects_created_[i] << std::endl;
                }
            }

            static void number_of_alive_collections_per_depth()
            {
                std::cout << "Number of alive collections per depth:" << std::endl;
                for(auto i = 0; i < objects_alive_.size(); i++)
                {
                    if(objects_alive_[i] > 0)
                        std::cout << " " << "Depth " << i << ": " << objects_alive_[i] << std::endl;
                }
            }
        };

        template <typename T> std::vector<int> Counter<T>::objects_created_(std::vector<int> (1000, 0));
        template <typename T> std::vector<int> Counter<T>::objects_alive_(std::vector<int> (1000, 0));
    }
}

#endif //PROJECT_COLLECTIONCOUNTER_HPP
