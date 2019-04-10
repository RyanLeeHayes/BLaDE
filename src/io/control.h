#ifndef IO_CONTROL_H
#define IO_CONTROL_H

// Forward declarations
class System;

void parse_if(char *line,System *system);
void parse_elseif(char *line,System *system);
void parse_else(char *line,System *system);
void parse_endif(char *line,System *system);
void parse_while(char *line,System *system);
void parse_endwhile(char *line,System *system);

#endif
