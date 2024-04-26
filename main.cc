#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <utility>
#include <vector>

double monte_carlo_integration(const std::pair<double, double>& x_borders,
                               const std::pair<double, double>& y_borders,
                               const std::function<double(double)>& function,
                               const unsigned int points_num,
                               const unsigned int num_threads) {
  const auto& [x_min, x_max] = x_borders;
  const auto& [y_min, y_max] = y_borders;
                            
  std::atomic<unsigned int> total_hits(0);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      std::mt19937 gen(std::random_device{}() + i);
      std::uniform_real_distribution<double> distribution_x(x_min, x_max);
      std::uniform_real_distribution<double> distribution_y(y_min, y_max);

      unsigned int localHits = 0;

      for (unsigned int j = 0; j < points_num / num_threads; ++j) {
        double x = distribution_x(gen);
        double y = distribution_y(gen);

        if ((0 <= y && y <= function(x)) || (function(x) <= y && y <= 0)) {
          localHits++;
        }
      }

      total_hits += localHits;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return (y_max - y_min) * (x_max - x_min) * 
         (static_cast<double>(total_hits) / points_num);
}

int main(int argc, char const* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <threads>" << std::endl;
    return 1;
  }

  const unsigned int thread_num = std::stoi(argv[1]);
  if (thread_num < 0) {
    std::cerr << "Number of threads must be non-negative." << std::endl;
    return 1;
  }

  std::pair<double, double> x_borders = {-1.0, 1.0};
  std::pair<double, double> y_borders = {0.0, 1.0};
  auto function = [](double x){ return std::sqrt(1 - x * x); };
  double expected_result = M_PI_2;
  double area = (y_borders.second - y_borders.first) *
                (x_borders.second - x_borders.first);

  double epsilon = 0.001;
  unsigned int bad_approximations = 0;
  unsigned int total_launches = 100;
  std::chrono::milliseconds total_time{0};

  unsigned int kPointsNum = 1000000;
  for (unsigned int i = 0; i < total_launches; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    double result = monte_carlo_integration(x_borders, y_borders, 
                                            function, kPointsNum, thread_num);

    auto end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (expected_result - result >= epsilon * area) {
      bad_approximations++;
    }
  }

  std::cout << "Average runtime of monte_carlo run: " << 
               static_cast<double>(total_time.count()) / total_launches <<
               " ms" << std::endl;

  double estimation = expected_result * (area - expected_result) / kPointsNum /
                      (epsilon * epsilon) / (area * area);

  if (static_cast<double>(bad_approximations) / total_launches <= estimation) {
    std::cout << "The estimation is correct" << std::endl;
  } else {
    std::cout << "Something went wrong..." << std::endl;
  }
}
