//
// Created by kades on 5/21/19.
//

#ifndef PROJECT_JACOBIAN_EQUATION_HPP
#define PROJECT_JACOBIAN_EQUATION_HPP

#include <vector>
#include <string>

#include <param_helper/json.hpp>

#include "../odesolver/util/header.hpp"
#include "../odesolver/util/dev_dat.hpp"

using json = nlohmann::json;

class JacobianWrapper
{
public:
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int row_idx, const int col_idx) = 0;
    
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int matrix_idx) = 0;
    
    virtual uint8_t get_dim() = 0;
    
    virtual json get_json() const {
        return {};
    }
};

struct JacobianEquation
{
    virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) = 0;
};

#endif //PROJECT_JACOBIAN_EQUATION_HPP
