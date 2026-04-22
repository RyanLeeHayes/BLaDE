#include <stdio.h>

#include "main/blade_log.h"
#include "system/system.h"
#include "io/control.h"
#include "io/variables.h"
#include "io/calculate.h"
#include "io/io.h"



// General functions
void finish_control(System *system)
{
  int level=system->control.size();
  int controlDepth=system->control[level-1].backtrace.size();
  int currentDepth=controlDepth;
  FILE *fp=system->control[level-1].fp;
  struct Frame frame;
  std::string previousType;
  char line[MAXLENGTHSTRING];

  while (currentDepth>=controlDepth) {
    fgetpos(fp,&frame.fp_pos);
    if (fgets(line, MAXLENGTHSTRING, fp) == NULL) {
      fatal(__FILE__,__LINE__,"File ended prematurely while searching for end of if or while construct\n");
    }
    // system->variables->substitute(line);
    frame.type=io_nexts(line);
    previousType=system->control[level-1].backtrace.back().type;
    if (frame.type=="if") {
      system->control[level-1].backtrace.push_back(frame);
    } else if (frame.type=="elseif") {
      if (previousType=="if" || previousType=="elseif") {
        system->control[level-1].backtrace.pop_back();
        system->control[level-1].backtrace.push_back(frame);
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected elseif\n");
      }
    } else if (frame.type=="else") {
      if (previousType=="if" || previousType=="elseif") {
        system->control[level-1].backtrace.pop_back();
        system->control[level-1].backtrace.push_back(frame);
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected else\n");
      }
    } else if (frame.type=="endif") {
      if (previousType=="if" || previousType=="elseif" || previousType=="else") {
        system->control[level-1].backtrace.pop_back();
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected endif\n");
      }
    } else if (frame.type=="while") {
      system->control[level-1].backtrace.push_back(frame);
    } else if (frame.type=="endwhile") {
      if (previousType=="while") {
        system->control[level-1].backtrace.pop_back();
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected endwhile\n");
      }
    }
    currentDepth=system->control[level-1].backtrace.size();
  }
}

// called when if evaluates to false
void find_next_if(System *system)
{
  int level=system->control.size();
  int controlDepth=system->control[level-1].backtrace.size();
  int currentDepth=controlDepth;
  FILE *fp=system->control[level-1].fp;
  struct Frame frame;
  std::string previousType;
  char line[MAXLENGTHSTRING];

  while (currentDepth>=controlDepth) {
    fgetpos(fp,&frame.fp_pos);
    if (fgets(line, MAXLENGTHSTRING, fp) == NULL) {
      fatal(__FILE__,__LINE__,"File ended prematurely while searching for next elseif, else, or endif construct\n");
    }
    // system->variables->substitute(line);
    frame.type=io_nexts(line);
    if (currentDepth==controlDepth) {
      if (frame.type=="elseif" || frame.type=="else" || frame.type=="endif") {
        // Backup a line and return
        fsetpos(fp,&frame.fp_pos);
        return;
      }
    }
    previousType=system->control[level-1].backtrace.back().type;
    if (frame.type=="if") {
      system->control[level-1].backtrace.push_back(frame);
    } else if (frame.type=="elseif") {
      if (previousType=="if" || previousType=="elseif") {
        system->control[level-1].backtrace.pop_back();
        system->control[level-1].backtrace.push_back(frame);
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected elseif\n");
      }
    } else if (frame.type=="else") {
      if (previousType=="if" || previousType=="elseif") {
        system->control[level-1].backtrace.pop_back();
        system->control[level-1].backtrace.push_back(frame);
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected else\n");
      }
    } else if (frame.type=="endif") {
      if (previousType=="if" || previousType=="elseif" || previousType=="else") {
        system->control[level-1].backtrace.pop_back();
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected endif\n");
      }
    } else if (frame.type=="while") {
      system->control[level-1].backtrace.push_back(frame);
    } else if (frame.type=="endwhile") {
      if (previousType=="while") {
        system->control[level-1].backtrace.pop_back();
      } else {
        fatal(__FILE__,__LINE__,"Found unexpected endwhile\n");
      }
    }
    currentDepth=system->control[level-1].backtrace.size();
  }
  fatal(__FILE__,__LINE__,"If statement terminated incorrectly somehow.\n");
}



// if functions
void parse_if(char *line,System *system)
{
  bool condition;
  int level=system->control.size();
  FILE *fp=system->control[level-1].fp;
  std::string token;
  fpos_t fp_pos;
  struct Frame frame;

  frame.type="if";
  frame.fp_pos=system->control[level-1].fp_pos;
  system->control[level-1].backtrace.push_back(frame);

  condition=(bool)variables_calculate(line);
  while (condition==false) {
    blade_log("if/elseif> evaluated as false");
    find_next_if(system);

    fgetpos(fp,&fp_pos);
    if (fgets(line, MAXLENGTHSTRING, fp) == NULL) {
      fatal(__FILE__,__LINE__,"File ended prematurely while searching for end of if construct\n");
    }
    system->variables->substitute(line);
    token=io_nexts(line);

    if (token=="elseif") {
      condition=(bool)variables_calculate(line);
    } else if (token=="else") {
      condition=true;
    } else if (token=="endif") {
      system->control[level-1].backtrace.pop_back();
      return;
    } else {
      fatal(__FILE__,__LINE__,"Bug in code, shouldn't reach this line. Sorry.\n");
    }
  }
  blade_log("if/elseif/else> evaluated as true");
}

void parse_elseif(char *line,System *system)
{
  int level=system->control.size();
  FILE *fp=system->control[level-1].fp;
  if (system->control[level-1].backtrace.size()==0) {
    fatal(__FILE__,__LINE__,"expecting to find elseif after if statement. Something went wrong\n");
  }
  struct Frame frame=system->control[level-1].backtrace.back();
  if (frame.type!="if" && frame.type!="elseif") {
    fatal(__FILE__,__LINE__,"expecting to find elseif after if statement. Something went wrong\n");
  }
  finish_control(system);
}

void parse_else(char *line,System *system)
{
  int level=system->control.size();
  FILE *fp=system->control[level-1].fp;
  if (system->control[level-1].backtrace.size()==0) {
    fatal(__FILE__,__LINE__,"expecting to find else after if statement. Something went wrong\n");
  }
  struct Frame frame=system->control[level-1].backtrace.back();
  if (frame.type!="if" && frame.type!="elseif") {
    fatal(__FILE__,__LINE__,"expecting to find else after if statement. Something went wrong\n");
  }
  finish_control(system);
}

void parse_endif(char *line,System *system)
{
  int level=system->control.size();
  FILE *fp=system->control[level-1].fp;
  if (system->control[level-1].backtrace.size()==0) {
    fatal(__FILE__,__LINE__,"expecting to find endif at end of if statement. Something went wrong\n");
  }
  struct Frame frame=system->control[level-1].backtrace.back();
  if (frame.type!="if" && frame.type!="elseif" && frame.type!="else") {
    fatal(__FILE__,__LINE__,"expecting to find endif at end of if statement. Something went wrong\n");
  }
  system->control[level-1].backtrace.pop_back();
}



// while functions
void evaluate_while(char *line,System *system)
{
  bool condition=(bool)variables_calculate(line);

  if (condition) {
    blade_log("while> evaluated as true");
  } else {
    blade_log("while> evaluated as false");
    finish_control(system);
  }
}

void parse_while(char *line,System *system)
{
  // Note position of while
  struct Frame frame;
  frame.type="while";
  int level=system->control.size();
  frame.fp_pos=system->control[level-1].fp_pos;
  system->control[level-1].backtrace.push_back(frame);

  evaluate_while(line,system);
}

void parse_endwhile(char *line,System *system)
{
  // Go back to while
  int level=system->control.size();
  FILE *fp=system->control[level-1].fp;
  if (system->control[level-1].backtrace.size()==0) {
    fatal(__FILE__,__LINE__,"expecting to find endwhile at end of while loop. Something went wrong\n");
  }
  struct Frame frame=system->control[level-1].backtrace.back();
  if (frame.type!="while") {
    fatal(__FILE__,__LINE__,"expecting to find endwhile at end of while loop. Something went wrong\n");
  }
  fsetpos(fp,&frame.fp_pos);
  fgets(line, MAXLENGTHSTRING, fp);
  system->variables->substitute(line);
  io_nexts(line);

  evaluate_while(line,system);
}
