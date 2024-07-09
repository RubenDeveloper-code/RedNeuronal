#ifndef __HANDLER_HPP__
#define __HANDLER_HPP__

#include <initializer_list>
#include <string>
namespace Handler {
void warning(std::initializer_list<std::string> args);
void terminalUserError(std::initializer_list<std::string> args);
void terminalSystemError(std::initializer_list<std::string> args);
} // namespace Handler

#endif
