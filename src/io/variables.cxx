#include <string.h>

#include "io/variables.h"
#include "io/calculate.h"
#include "io/io.h"
#include "system/system.h"

#include "main/defines.h"
#include "main/blade_log.h"

#ifdef REPLICAEXCHANGE
#include <mpi.h>
#endif



// Class constructors
Variables::Variables()
{
  setup_parse_variables();

#ifdef REPLICAEXCHANGE
  int rank;
  char rankstring[10];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sprintf(rankstring,"%d",rank);
  data["rexrank"]=rankstring;
#endif
}

Variables::~Variables()
{
}



// Parsing functions
void Variables::setup_parse_variables()
{
  parseVariables[""]=&Variables::error;
  helpVariables[""]="?> How did we get here?\n";
  parseVariables["reset"]=&Variables::reset;
  helpVariables["reset"]="?variables reset> This deletes the data stored in varaibles.\n";
  parseVariables["set"]=&Variables::set;
  helpVariables["set"]="?variables set token value> This saves the value value into token, so value can later be accessed with the {token} syntax using curly brackets. Values are stored as strings, but can be treated as numbers using the calculate command.\n";
  parseVariables["setupper"]=&Variables::setupper;
  helpVariables["setupper"]="?variables setupper token value> Casts value to uppercase, and then saves the result into token\n";
  parseVariables["setlower"]=&Variables::setlower;
  helpVariables["setlower"]="?variables setlower token value> Casts value to lowercase, and then saves the result into token\n";
  parseVariables["calculate"]=&Variables::calculate;
  helpVariables["calculate"]="?variables calculate token int|real expression> Calculate the result of expression, cast it into a int or real formatted string, and store that string in token. The systax parsing for expressions is not very advanced. Consequently, operators are always first, which means there is no need to define and order of operations. sin(x+y) is expressed as \"sin + {x} {y}\", while sin(x)+y is \"+ sin {x} {y}\". Currently supported operators are in flux, but should soon include + - * / sin cos sqrt log exp floor\n";
  parseVariables["print"]=&Variables::dump;
  helpVariables["print"]="?variables print> This prints all variable token/value pairs to standard out\n";
  parseVariables["help"]=&Variables::help;
  helpVariables["help"]="?variables help [directive]> Prints help on variables directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void parse_variables(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  io_nexta(line,token);
  name=token;
  if (system->variables->parseVariables.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->variables->parseVariables[name]
  // within the object: system->variables
  // with arguments: (line,token,system)
  (system->variables->*(system->variables->parseVariables[name]))(line,token,system);
}



// Parsing helper functions
void Variables::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after variables: %s\n",token);
}

void Variables::reset(char *line,char *token,System *system)
{
  data.clear();
}

void Variables::set(char *line,char *token,System *system)
{
  std::string key,value;

  key=io_nexts(line);

  value=io_nexts(line);
  while (io_peeks(line)!="") {
    value.append(" ");
    value.append(io_nexts(line));
  }

  data[key]=value;
}

void Variables::setupper(char *line,char *token,System *system)
{
  std::string key,value;

  key=io_nexts(line);

  value=io_uppers(io_nexts(line));
  while (io_peeks(line)!="") {
    value.append(" ");
    value.append(io_uppers(io_nexts(line)));
  }

  data[key]=value;
}

void Variables::setlower(char *line,char *token,System *system)
{
  std::string key,value;

  key=io_nexts(line);

  value=io_lowers(io_nexts(line));
  while (io_peeks(line)!="") {
    value.append(" ");
    value.append(io_lowers(io_nexts(line)));
  }

  data[key]=value;
}

void Variables::calculate(char *line,char *token,System *system)
{
  char value[MAXLENGTHSTRING];

  std::string key=io_nexts(line);
  std::string type=io_nexts(line);
  if (type=="int") {
    sprintf(value,"%d",(int)floor(variables_calculate(line)+0.5));
  } else if (type=="real") {
    sprintf(value,"%g",variables_calculate(line));
  } else {
    fatal(__FILE__,__LINE__,"Specified %d for data formatting. Use int or real\n");
  }
  data[key]=value;
}

void Variables::dump(char *line,char *token,System *system)
{
  char buf[256];
  for (std::map<std::string,std::string>::iterator ii=data.begin(); ii!=data.end(); ii++) {
    snprintf(buf, sizeof(buf), "variables print> %s = %s\n", ii->first.c_str(), ii->second.c_str());
    blade_log(buf);
  }
}

void Variables::help(char *line,char *token,System *system)
{
  char name[MAXLENGTHSTRING];
  char buf[256];
  io_nexta(line,name);
  if (name=="") {
    blade_log("?variables> Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpVariables.begin(); ii!=helpVariables.end(); ii++) {
      snprintf(buf, sizeof(buf), " %s", ii->first.c_str());
      blade_log(buf);
    }
    blade_log("\n");
  } else if (helpVariables.count(name)==1) {
    blade_log(helpVariables[name].c_str());
  } else {
    error(line,name,system);
  }
}



// Other functions
void Variables::substitute(char *line)
{
  int openBrace,closeBrace=0;
  int i;
  int lineLength=strlen(line);
  std::string token,value;
  int tokenLength,valueLength;
  bool anySubstitutions=false;

  while (closeBrace>=0) {
    openBrace=-1;
    closeBrace=-1;
    for (i=0; (line[i]!='\0' && line[i]!='!') && i<MAXLENGTHSTRING; i++) {
      if (line[i]=='{') {
        openBrace=i;
      } else if (line[i]=='}') {
        closeBrace=i;
        break;
      }
    }
    if ((openBrace==-1) != (closeBrace==-1)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "DEBUG: openBrace %d closeBrace %d\n", openBrace, closeBrace);
      blade_log(buf);
      fatal(__FILE__,__LINE__,"Mismatched curly braces in string \"%s\"\n",line);
    }

    if (closeBrace>=0) {
      // null terminate token
      line[closeBrace]='\0';
      token=&line[openBrace+1];
      line[closeBrace]='}';

      // get value
      if (data.count(token)==0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "DEBUG: openBrace %d closeBrace %d\n", openBrace, closeBrace);
        blade_log(buf);
        fatal(__FILE__,__LINE__,"Unrecognized variable token name {%s} in line \"%s\"\n",token.c_str(),line);
      }
      value=data[token];

      tokenLength=token.length();
      valueLength=value.length();
      lineLength+=valueLength-tokenLength-2;
      if (lineLength+1>=MAXLENGTHSTRING) {
        fatal(__FILE__,__LINE__,"Max string length %d exceeded on line \"%s\"\n",MAXLENGTHSTRING,line);
      }
      io_shift(&line[openBrace],tokenLength+2);
      io_shift(&line[openBrace],-valueLength);
      for (i=0; i<valueLength; i++) {
        line[i+openBrace]=value.c_str()[i];
      }
      anySubstitutions=true;
    }
  }
  if (anySubstitutions) {
    char buf[256];
    snprintf(buf, sizeof(buf), "SUBSTITUTE> %s", line);
    blade_log(buf);
  }
}
