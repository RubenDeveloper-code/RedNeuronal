#include <cmath>
#include <iostream>
// naaaa
class Graph {
    public:
      Graph(int x, int y, bool _dinamic)
          : maxX(x), maxY(y), dinamic(_dinamic) {}
      void drawGraph(float x, float y) {
            std::cout << "1"
                      << "\n";
            float precision = 0.1;
            float _y = std::round((float)y / precision) * precision;
            float _x = std::round((float)x / precision) * precision;
            std::cout << _x << " " << _y << "\n";
            for (int i = 0; i < 9; i++) {
                  std::cout << " |";
                  for (int j = 0; j < 20; j++) {
                        if (std::fabs(((10.0 - j) / 10.0) - _y) < 1e-2 &&
                            std::fabs((((10.0 - i) * 2) / 10.0) - _x) < 1e-2)
                              std::cout << "*";
                        else
                              std::cout << " ";
                  }
                  std::cout << "\n";
            }
            std::cout << "0|";
            for (int j = 0; j < 20; j++) {
                  std::cout << "_";
            }
            std::cout << "1"
                      << "\n";
      }

    private:
      int maxX, maxY;
      bool dinamic;
};

int main() {
      Graph g(10, 20, true);
      g.drawGraph(0.42, 0.17);
}
