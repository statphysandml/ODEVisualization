//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_PATH_PARAMETERS_HPP
#define PROGRAM_PATH_PARAMETERS_HPP

struct PathParameters
{
    PathParameters(const std::string theory_,
                   const std::string mode_type_,
                   const std::string root_dir_="/data/",
                   const bool relative_path_=true) :
            theory(theory_), mode_type(mode_type_), root_dir(root_dir_), relative_path(relative_path_)
    {
        written_to_file = false;
    }

    std::string get_base_path() const
    {
        return root_dir + "/" + theory + "/";
    }

    const std::string theory;
    const std::string mode_type;
    const std::string root_dir;
    const bool relative_path;
    bool written_to_file;
};

#endif //PROGRAM_PATH_PARAMETERS_HPP
