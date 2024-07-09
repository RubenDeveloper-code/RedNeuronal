#include "../../include/alerts/handler.hpp"
#include <cstdlib>
#include <initializer_list>
#include <iostream>

inline void printList(std::initializer_list<std::string> args) {
      for (auto arg : args) {
            std::cout << arg;
      }
}
void Handler::warning(std::initializer_list<std::string> args) {
      std::cout << "warning: ";
      printList(args);
      std::cout << "\n";
}
void Handler::terminalUserError(std::initializer_list<std::string> args) {
      std::cout << "User Error: ";
      printList(args);
      exit(1);
}
void Handler::terminalSystemError(std::initializer_list<std::string> args) {
      std::cout << "System Error: ";
      printList(args);
      exit(2);
}
