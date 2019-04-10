#include <stdio.h>

#include <string>

#include "io/calculate.h"
#include "io/io.h"

// Calculate function
real variables_calculate(char *line)
{
  std::string token=io_nexts(line);

  if (token=="+") {
    return variables_calculate(line) + variables_calculate(line);
  } else if (token=="-") {
    return variables_calculate(line) - variables_calculate(line);
  } else if (token=="*") {
    return variables_calculate(line) * variables_calculate(line);
  } else if (token=="/") {
    return variables_calculate(line) / variables_calculate(line);
  } else if (token=="==") {
    return variables_calculate(line) == variables_calculate(line);
  } else if (token==">=") {
    return variables_calculate(line) >= variables_calculate(line);
  } else if (token=="<=") {
    return variables_calculate(line) <= variables_calculate(line);
  } else if (token==">") {
    return variables_calculate(line) > variables_calculate(line);
  } else if (token=="<") {
    return variables_calculate(line) < variables_calculate(line);
  } else if (token=="&&") {
    return variables_calculate(line) && variables_calculate(line);
  } else if (token=="||") {
    return variables_calculate(line) || variables_calculate(line);
  } else if (token=="not") {
    return variables_calculate(line)==0;
  } else if (token=="sqrt") {
    return sqrt(variables_calculate(line));
  } else if (token=="exp") {
    return exp(variables_calculate(line));
  } else if (token=="log") {
    return log(variables_calculate(line));
  } else if (token=="sin") {
    return sin(variables_calculate(line));
  } else if (token=="cos") {
    return cos(variables_calculate(line));
  } else if (token=="tan") {
    return tan(variables_calculate(line));
  } else if (token=="floor") {
    return floor(variables_calculate(line));
  } else if (token=="erf") {
    return erf(variables_calculate(line));
  } else if (token=="eq") {
    return io_nexts(line) == io_nexts(line);
  } else if (token=="pi") {
    return M_PI;
  } else {
    // try converting to a number
    double number;
    int read=sscanf(token.c_str(),"%lg",&number);
    if (read==1) {
      return (real) number;
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token (%s) in calculate statement\n",token.c_str());
    }
  }
}
