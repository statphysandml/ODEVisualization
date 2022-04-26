//
// Created by kades on 5/21/19.
//

#ifndef PROJECT_JACOBIAN_EQUATION_HPP
#define PROJECT_JACOBIAN_EQUATION_HPP

#include <vector>
#include <string>

#include "../odesolver/util/header.hpp"
#include "../odesolver/util/dev_dat.hpp"

class JacobianWrapper
{
public:
    static JacobianWrapper * make_jacobian(std::string theory);

    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int row_idx, const int col_idx) = 0;
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int matrix_idx) = 0;
    virtual uint8_t get_dim() = 0;
    static std::string name()
    {
        return "JacobianWrapper";
    }
};

struct JacobianEquation
{
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) = 0;
};

#endif //PROJECT_JACOBIAN_EQUATION_HPP
