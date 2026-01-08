#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXf A(2, 2);
  A << 1, 2, 3, 4;
  std::cout << "A:\n" << A << "\n";
  return 0;
}