// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <functional>

#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam_points/optimizers/gaussian_factor_graph_solver.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_optimization_status.hpp>

namespace gtsam_points {

class LevenbergMarquardtExtParams : public gtsam::LevenbergMarquardtParams {
public:
  LevenbergMarquardtExtParams() : gtsam::LevenbergMarquardtParams() {}

  static LevenbergMarquardtExtParams EnsureHasOrdering(LevenbergMarquardtExtParams params, const gtsam::NonlinearFactorGraph& graph) {
    if (!params.ordering) params.ordering = gtsam::Ordering::Create(params.orderingType, graph);
    return params;
  }

  static LevenbergMarquardtExtParams ReplaceOrdering(LevenbergMarquardtExtParams params, const gtsam::Ordering& ord) {
    params.ordering = ord;
    return params;
  }

  void set_verbose() {
    callback = [](const LevenbergMarquardtOptimizationStatus& status, const gtsam::Values&) { std::cout << status.to_string() << std::endl; };
  }

  std::shared_ptr<GaussianFactorGraphSolver> solver;

  std::function<bool(const gtsam::Values& values)> termination_criteria;
  std::function<void(const LevenbergMarquardtOptimizationStatus&, const gtsam::Values&)> callback;  // callback for optimization iteration
  std::function<void(const std::string&)> status_msg_callback;
};

}  // namespace gtsam_points
