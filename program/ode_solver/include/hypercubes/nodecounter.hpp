//
// Created by lukas on 13.03.19.
//

#ifndef PROJECT_NODECOUNTER_HPP
#define PROJECT_NODECOUNTER_HPP

#include <vector>
#include <iostream>

// https://stackoverflow.com/questions/1926605/how-to-count-the-number-of-objects-created-in-c
template <typename T>
struct NodeCounter
{
    NodeCounter(const int depth)
    {
        objects_created[depth]++;
        objects_alive[depth]++;
    }

    static std::vector<int> objects_created;
    static std::vector<int> objects_alive;

    static void print_statistics()
    {
        number_of_created_nodes_per_depth();
        number_of_alive_nodes_per_depth();
    }

    static void number_of_created_nodes_per_depth()
    {
        std::cout << "Number of created nodes per depth:" << std::endl;
        for(auto i = 0; i < objects_created.size(); i++)
        {
            if(objects_created[i] > 0)
                std::cout << " " << "Depth " << i << ": " << objects_created[i] << std::endl;
        }
    }

    static void number_of_alive_nodes_per_depth()
    {
        std::cout << "Number of alive nodes per depth:" << std::endl;
        for(auto i = 0; i < objects_alive.size(); i++)
        {
            if(objects_alive[i] > 0)
                std::cout << " " << "Depth " << i << ": " << objects_alive[i] << std::endl;
        }
    }
};

template <typename T> std::vector<int> NodeCounter<T>::objects_created( std::vector<int> (1000, 0) );
template <typename T> std::vector<int> NodeCounter<T>::objects_alive( std::vector<int> (1000, 0) );

#endif //PROJECT_NODECOUNTER_HPP
