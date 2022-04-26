//
// Created by lukas on 31.03.20.
//

#ifndef PROGRAM_PATH_PARAMETERS_HPP
#define PROGRAM_PATH_PARAMETERS_HPP

struct PathParameters
{
    PathParameters(const std::string theory,
                   const std::string mode_type,
                   const std::string root_dir="/data/",
                   const bool relative_path=true) :
            theory_(theory), mode_type_(mode_type), root_dir_(root_dir), relative_path_(relative_path)
    {
        written_to_file_ = false;
    }

    std::string get_base_path() const
    {
        return root_dir_ + "/" + theory_ + "/";
    }

    const std::string theory_;
    const std::string mode_type_;
    const std::string root_dir_;
    const bool relative_path_;
    bool written_to_file_;
};

#endif //PROGRAM_PATH_PARAMETERS_HPP
