#include "../../include/alerts/messages.hpp"
#include <iostream>
inline void printList(std::initializer_list<std::string> args) {
      for (auto arg : args) {
            std::cout << arg;
      }
}
void Messages::Message(std::initializer_list<std::string> args) {
      printList(args);
      std::cout << "\n";
}
bool Messages::Confirmation(std::initializer_list<std::string> args) {
      printList(args);
      std::cout << "y/n: ";
      char input = std::cin.get();
      if (input == 'y' || input == 'Y')
            return true;
      return false;
}
