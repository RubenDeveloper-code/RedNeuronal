#ifndef __MESSAGES_HPP__
#define __MESSAGES_HPP__

#include <initializer_list>
#include <string>
namespace Messages {
void Message(std::initializer_list<std::string> args);
bool Confirmation(std::initializer_list<std::string> args);
} // namespace Messages

#endif
