#include <ctype.h>
#include <math.h>
#include <map>
#include <set>
#include <string>
#include <string.h>
#include <vector>

#include "drude/drude_plugin.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/system.h"
#include "io/io.h"

static inline std::string drude_upper_copy(const std::string &text)
{
  std::string out=text;
  for (size_t i=0; i<out.size(); i++) {
    out[i]=(char)toupper((unsigned char)out[i]);
  }
  return out;
}

static inline std::string drude_make_residue_key(const AtomStructure &at)
{
  return at.segName+"|"+at.resIdx+"|"+at.resName;
}

static inline bool drude_is_drude_candidate(const AtomStructure &at)
{
  std::string atomName=drude_upper_copy(at.atomName);
  std::string atomType=drude_upper_copy(at.atomTypeName);

  // Lone-pair virtual sites are not Drude particles and must not be treated as spring candidates.
  if (atomName.find("LP")==0 || atomType.find("LP")==0) return false;

  if (atomName.find("D")==0) return true;
  if (atomType.find("D")==0) return true;

  return false;
}

static inline void drude_add_edge(std::vector< std::set<int> > &graph,int a,int b)
{
  if (a==b) return;
  graph[a].insert(b);
  graph[b].insert(a);
}

static inline void drude_add_exclusion(std::vector< std::set<int> > &excl,int a,int b)
{
  if (a==b) return;
  excl[a].insert(b);
  excl[b].insert(a);
}

static inline long long drude_ordered_pair_key(int a,int b)
{
  int lo=(a<b? a : b);
  int hi=(a<b? b : a);
  return (((long long)lo)<<32)|(unsigned int)hi;
}

static std::map<int,real> drude_collect_spring_k_by_drude(const DrudePlugin *plugin)
{
  std::map<int,real> springKByDrude;
  for (size_t i=0; i<plugin->springPairs_tmp.size(); i++) {
    springKByDrude[plugin->springPairs_tmp[i].idx[0]]=plugin->springPairs_tmp[i].k;
  }
  return springKByDrude;
}

static real drude_autobuild_polarizability(System *system,
  const std::map<int,real> &springKByDrude,int d,int p)
{
  Structure *struc=system->structure;
  DrudePlugin *plugin=system->drudePlugin;

  std::map<int,real>::const_iterator it=plugin->explicitPolarizabilities.find(d);
  if (it!=plugin->explicitPolarizabilities.end()) return it->second;
  it=plugin->explicitPolarizabilities.find(p);
  if (it!=plugin->explicitPolarizabilities.end()) return it->second;

  if (struc->atomList[p].polarizability>(real)0) return struc->atomList[p].polarizability;
  if (struc->atomList[d].polarizability>(real)0) return struc->atomList[d].polarizability;

  it=springKByDrude.find(d);
  if (it==springKByDrude.end() || !isfinite((double)it->second) || it->second<=(real)0) {
    fatal(__FILE__,__LINE__,
      "drude autobuild cannot derive polarizability for drude atom %d parent %d: missing positive spring k\n",
      d,p);
  }
  real q=struc->atomList[d].charge;
  real alpha=fabs((double)(kELECTRIC*q*q/it->second));
  if (!isfinite((double)alpha) || alpha<=(real)0) {
    fatal(__FILE__,__LINE__,
      "drude autobuild derived invalid polarizability for drude atom %d parent %d: q=%g k=%g alpha=%g\n",
      d,p,(double)q,(double)it->second,(double)alpha);
  }
  return alpha;
}

static real drude_autobuild_atom_thole(System *system,int d,int p)
{
  Structure *struc=system->structure;
  DrudePlugin *plugin=system->drudePlugin;

  std::map<int,real>::const_iterator it=plugin->explicitTholes.find(d);
  if (it!=plugin->explicitTholes.end()) return it->second;
  it=plugin->explicitTholes.find(p);
  if (it!=plugin->explicitTholes.end()) return it->second;

  if (struc->atomList[p].thole>(real)0) return struc->atomList[p].thole;
  if (struc->atomList[d].thole>(real)0) return struc->atomList[d].thole;
  return plugin->defaultThole;
}

static real drude_autobuild_screening_scale(System *system,
  const std::map<int,real> &springKByDrude,int d1,int p1,int d2,int p2,
  bool usePairThole,real pairThole)
{
  real alpha1=drude_autobuild_polarizability(system,springKByDrude,d1,p1);
  real alpha2=drude_autobuild_polarizability(system,springKByDrude,d2,p2);
  real thole=usePairThole? pairThole :
    drude_autobuild_atom_thole(system,d1,p1)+drude_autobuild_atom_thole(system,d2,p2);

  if (!isfinite((double)thole) || thole<(real)0) {
    fatal(__FILE__,__LINE__,
      "drude autobuild resolved invalid Thole value for dipoles (%d,%d)-(%d,%d): %g\n",
      d1,p1,d2,p2,(double)thole);
  }
  if (thole==(real)0) return (real)0;

  real alphaProduct=alpha1*alpha2;
  if (!isfinite((double)alphaProduct) || alphaProduct<=(real)0) {
    fatal(__FILE__,__LINE__,
      "drude autobuild resolved invalid alpha product for dipoles (%d,%d)-(%d,%d): %g * %g\n",
      d1,p1,d2,p2,(double)alpha1,(double)alpha2);
  }
  return thole/(real)pow((double)alphaProduct,1.0/6.0);
}

static void drude_autobuild_add_nbthole_atom_pair(std::vector<DrudeNBTholePairPotential> &out,
  Structure *struc,int a,int b,real screeningScale)
{
  if (a==b) return;

  DrudeNBTholePairPotential pp;
  pp.idx[0]=a;
  pp.idx[1]=b;
  pp.screeningScale=screeningScale;
  pp.energyScale=kELECTRIC*struc->atomList[a].charge*struc->atomList[b].charge;
  out.push_back(pp);
}

static void drude_autobuild_add_nbthole_all_subpairs(std::vector<DrudeNBTholePairPotential> &out,
  Structure *struc,int d1,int p1,int d2,int p2,real screeningScale)
{
  int first[4]={p1,p1,d1,d1};
  int second[4]={p2,d2,p2,d2};
  for (int i=0; i<4; i++) {
    drude_autobuild_add_nbthole_atom_pair(out,struc,first[i],second[i],screeningScale);
  }
}

static void drude_autobuild_add_nbthole_14_main_subpairs(std::vector<DrudeNBTholePairPotential> &out,
  Structure *struc,int d1,int p1,int d2,int p2,real screeningScale)
{
  // OpenMM excludes only the parent-parent 1-4 pair from the main nbtforce.
  drude_autobuild_add_nbthole_atom_pair(out,struc,p1,d2,screeningScale);
  drude_autobuild_add_nbthole_atom_pair(out,struc,d1,p2,screeningScale);
  drude_autobuild_add_nbthole_atom_pair(out,struc,d1,d2,screeningScale);
}

static void drude_autobuild_detect_pairs(System *system,std::vector<int2> &pairsOut)
{
  Structure *struc=system->structure;
  int atomCount=struc->atomCount;
  std::vector< std::set<int> > bondGraph(atomCount);
  std::vector<bool> isDrude(atomCount,false);
  std::map<std::string,std::vector<int> > atomsByResidue;

  for (int i=0; i<atomCount; i++) {
    isDrude[i]=drude_is_drude_candidate(struc->atomList[i]);
    atomsByResidue[drude_make_residue_key(struc->atomList[i])].push_back(i);
  }
  for (size_t i=0; i<struc->bondList.size(); i++) {
    int a=struc->bondList[i].i[0];
    int b=struc->bondList[i].i[1];
    if (a<0 || a>=atomCount || b<0 || b>=atomCount) {
      fatal(__FILE__,__LINE__,"drude autobuild found out-of-range bond (%d,%d), atomCount=%d\n",a,b,atomCount);
    }
    drude_add_edge(bondGraph,a,b);
  }

  std::set<long long> uniquePairs;
  pairsOut.clear();
  for (int d=0; d<atomCount; d++) {
    if (!isDrude[d]) continue;

    int parent=-1;
    std::vector<int> bondParents;
    for (std::set<int>::const_iterator it=bondGraph[d].begin(); it!=bondGraph[d].end(); it++) {
      if (!isDrude[*it]) bondParents.push_back(*it);
    }
    if (bondParents.size()==1) {
      parent=bondParents[0];
    } else if (bondParents.size()>1) {
      fatal(__FILE__,__LINE__,
        "drude autobuild found ambiguous bonded parent candidates for drude atom %d (%s): %d candidates\n",
        d,struc->atomList[d].atomName.c_str(),(int)bondParents.size());
    } else {
      std::string key=drude_make_residue_key(struc->atomList[d]);
      if (atomsByResidue.count(key)!=1) {
        fatal(__FILE__,__LINE__,
          "drude autobuild internal residue lookup failed for atom %d residue key %s\n",
          d,key.c_str());
      }
      const std::vector<int> &residueAtoms=atomsByResidue[key];
      std::vector<int> candidates;
      for (size_t i=0; i<residueAtoms.size(); i++) {
        int a=residueAtoms[i];
        if (a==d) continue;
        if (isDrude[a]) continue;
        candidates.push_back(a);
      }
      if (candidates.size()==1) {
        parent=candidates[0];
      } else if (candidates.size()>1) {
        real bestMass=(real)-1;
        int bestAtom=-1;
        int bestCount=0;
        for (size_t i=0; i<candidates.size(); i++) {
          int a=candidates[i];
          real mass=struc->atomList[a].mass;
          if (mass>bestMass+(real)1e-8) {
            bestMass=mass;
            bestAtom=a;
            bestCount=1;
          } else if (fabs((double)(mass-bestMass))<1e-8) {
            bestCount++;
          }
        }
        if (bestCount==1 && bestAtom>=0) {
          parent=bestAtom;
        } else {
          fatal(__FILE__,__LINE__,
            "drude autobuild found ambiguous residue parent candidates for drude atom %d in residue %s\n",
            d,key.c_str());
        }
      } else {
        fatal(__FILE__,__LINE__,
          "drude autobuild cannot find parent for drude atom %d (%s). "
          "Provide explicit drude spring pairs or PSF bonds.\n",
          d,struc->atomList[d].atomName.c_str());
      }
    }

    if (parent<0 || parent>=atomCount || parent==d) {
      fatal(__FILE__,__LINE__,
        "drude autobuild resolved invalid parent for drude atom %d: parent=%d\n",
        d,parent);
    }
    long long key=(((long long)d)<<32)|(unsigned int)parent;
    if (uniquePairs.count(key)>0) continue;
    uniquePairs.insert(key);
    pairsOut.push_back(make_int2(d,parent));
  }

  if (pairsOut.empty()) {
    fatal(__FILE__,__LINE__,
      "drude autobuild found zero drude-parent pairs. "
      "Check PSF atom naming/masses or provide manual drude spring directives.\n");
  }
}

static void drude_autobuild_build_springs(System *system,const std::vector<int2> &pairs)
{
  Structure *struc=system->structure;
  Parameters *param=system->parameters;
  DrudePlugin *plugin=system->drudePlugin;

  plugin->springPairs_tmp.clear();
  for (size_t i=0; i<pairs.size(); i++) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    TypeName2 type;
    TypeName2 fallbackDX;
    TypeName2 fallbackXD;
    type.t[0]=struc->atomList[d].atomTypeName;
    type.t[1]=struc->atomList[p].atomTypeName;
    fallbackDX.t[0]=type.t[0];
    fallbackDX.t[1]="X";
    fallbackXD.t[0]="X";
    fallbackXD.t[1]=type.t[0];
    std::map<TypeName2,struct BondParameter>::const_iterator bpIt=param->bondParameter.find(type);
    if (bpIt==param->bondParameter.end()) {
      // Fall back to generic Drude spring definitions (for example X-DRUD) used by CHARMM Drude force fields.
      bpIt=param->bondParameter.find(fallbackDX);
      if (bpIt==param->bondParameter.end()) {
        bpIt=param->bondParameter.find(fallbackXD);
      }
    }
    if (bpIt==param->bondParameter.end()) {
      fatal(__FILE__,__LINE__,
        "drude autobuild missing BOND parameter for drude-parent types (%s,%s) at atoms (%d,%d). "
        "fallback(%s,X)=%d fallback(X,%s)=%d\n",
        type.t[0].c_str(),type.t[1].c_str(),d,p,
        type.t[0].c_str(),(int)param->bondParameter.count(fallbackDX),
        type.t[0].c_str(),(int)param->bondParameter.count(fallbackXD));
    }
    const BondParameter &bp=bpIt->second;
    if (!isfinite((double)bp.kb) || bp.kb<=(real)0) {
      fatal(__FILE__,__LINE__,
        "drude autobuild resolved invalid spring constant kb=%g for types (%s,%s)\n",
        (double)bp.kb,type.t[0].c_str(),type.t[1].c_str());
    }

    DrudeSpringPairPotential pp;
    pp.idx[0]=d;
    pp.idx[1]=p;
    drude_spring_clear_anisotropy(&pp);
    pp.k=bp.kb;
    pp.r0=(real)0;
    std::map<int,DrudeAnisotropyStructure>::const_iterator anisoIt=
      struc->drudeAnisotropyByParent.find(p);
    if (anisoIt!=struc->drudeAnisotropyByParent.end()) {
      pp.anisoIdx[0]=anisoIt->second.axis1Idx;
      pp.anisoIdx[1]=anisoIt->second.axis3Idx;
      pp.anisoIdx[2]=anisoIt->second.axis4Idx;
      pp.aniso12=anisoIt->second.aniso12;
      pp.aniso34=anisoIt->second.aniso34;
    }
    plugin->springPairs_tmp.push_back(pp);
  }
}

static std::vector<int2> drude_collect_effective_springs(const DrudePlugin *plugin)
{
  std::vector<int2> pairs;
  pairs.reserve(plugin->springPairs_tmp.size());
  for (size_t i=0; i<plugin->springPairs_tmp.size(); i++) {
    int d=plugin->springPairs_tmp[i].idx[0];
    int p=plugin->springPairs_tmp[i].idx[1];
    pairs.push_back(make_int2(d,p));
  }
  return pairs;
}

static void drude_autobuild_build_screened(System *system,const std::vector<int2> &pairs)
{
  Structure *struc=system->structure;
  DrudePlugin *plugin=system->drudePlugin;
  int atomCount=struc->atomCount;
  std::vector< std::set<int> > parentGraph(atomCount);
  std::vector<bool> isSpringDrude(atomCount,false);
  std::map<int,real> springKByDrude=drude_collect_spring_k_by_drude(plugin);
  std::map<int,std::vector<size_t> > pairIndicesByParent;

  for (size_t i=0; i<pairs.size(); i++) {
    int d=pairs[i].x;
    if (d>=0 && d<atomCount) isSpringDrude[d]=true;
    pairIndicesByParent[pairs[i].y].push_back(i);
  }

  for (size_t i=0; i<struc->bondList.size(); i++) {
    int a=struc->bondList[i].i[0];
    int b=struc->bondList[i].i[1];
    if (a<0 || a>=atomCount || b<0 || b>=atomCount) {
      fatal(__FILE__,__LINE__,"drude autobuild screened found out-of-range bond (%d,%d), atomCount=%d\n",a,b,atomCount);
    }
    if (isSpringDrude[a] || isSpringDrude[b]) continue;
    drude_add_edge(parentGraph,a,b);
  }

  plugin->screenedPairs_tmp.clear();
  for (size_t i=0; i<pairs.size(); i++) {
    int d1=pairs[i].x;
    int p1=pairs[i].y;
    std::set<int> candidateParents;
    // Screened Drude pairs only exist for parent atoms separated by one or two bonds.
    for (std::set<int>::const_iterator it1=parentGraph[p1].begin();
         it1!=parentGraph[p1].end(); it1++) {
      int p2=*it1;
      candidateParents.insert(p2);
      for (std::set<int>::const_iterator it2=parentGraph[p2].begin();
           it2!=parentGraph[p2].end(); it2++) {
        if (*it2!=p1) candidateParents.insert(*it2);
      }
    }

    for (std::set<int>::const_iterator pit=candidateParents.begin();
         pit!=candidateParents.end(); pit++) {
      std::map<int,std::vector<size_t> >::const_iterator listIt=pairIndicesByParent.find(*pit);
      if (listIt==pairIndicesByParent.end()) continue;
      const std::vector<size_t> &neighbors=listIt->second;
      for (size_t jj=0; jj<neighbors.size(); jj++) {
        size_t j=neighbors[jj];
        if (j<=i) continue;
        int d2=pairs[j].x;
        int p2=pairs[j].y;

        DrudeScreenedPairPotential pp;
        pp.idx[0]=d1;
        pp.idx[1]=p1;
        pp.idx[2]=d2;
        pp.idx[3]=p2;
        pp.screeningScale=drude_autobuild_screening_scale(system,springKByDrude,d1,p1,d2,p2,false,(real)0);
        pp.energyScale=kELECTRIC*struc->atomList[d1].charge*struc->atomList[d2].charge;
        plugin->screenedPairs_tmp.push_back(pp);
      }
    }
  }
}

static void drude_autobuild_build_nbthole(System *system,const std::vector<int2> &pairs)
{
  Structure *struc=system->structure;
  Parameters *param=system->parameters;
  DrudePlugin *plugin=system->drudePlugin;
  int atomCount=struc->atomCount;
  std::vector< std::set<int> > bondExcl(atomCount);
  std::vector< std::set<int> > angleExcl(atomCount);
  std::vector< std::set<int> > diheExcl(atomCount);
  std::vector< std::set<int> > allExcl(atomCount);
  std::vector<int2> structureBonds;
  std::vector<bool> isSpringDrude(atomCount,false);
  std::set<long long> screenedDrudePairs;
  std::map<int,real> springKByDrude=drude_collect_spring_k_by_drude(plugin);
  std::map<std::string,std::vector<size_t> > pairIndicesByParentType;

  for (size_t i=0; i<pairs.size(); i++) {
    int d=pairs[i].x;
    if (d>=0 && d<atomCount) isSpringDrude[d]=true;
  }

  for (size_t i=0; i<struc->bondList.size(); i++) {
    int a=struc->bondList[i].i[0];
    int b=struc->bondList[i].i[1];
    if (a<0 || a>=atomCount || b<0 || b>=atomCount) {
      fatal(__FILE__,__LINE__,"drude autobuild nbthole found out-of-range bond (%d,%d), atomCount=%d\n",a,b,atomCount);
    }
    if (isSpringDrude[a] || isSpringDrude[b]) continue;
    structureBonds.push_back(make_int2(a,b));
    if (allExcl[a].count(b)==0) {
      drude_add_exclusion(bondExcl,a,b);
      drude_add_exclusion(allExcl,a,b);
    }
  }

  for (size_t i=0; i<structureBonds.size(); i++) {
    int ab[2]={structureBonds[i].x,structureBonds[i].y};
    for (int j=0; j<2; j++) {
      int from=ab[j];
      int through=ab[1-j];
      for (std::set<int>::const_iterator it=bondExcl[through].begin(); it!=bondExcl[through].end(); it++) {
        int to=*it;
        if (to==from) continue;
        if (allExcl[from].count(to)==0) {
          drude_add_exclusion(angleExcl,from,to);
          drude_add_exclusion(allExcl,from,to);
        }
      }
    }
  }

  for (size_t i=0; i<structureBonds.size(); i++) {
    int ab[2]={structureBonds[i].x,structureBonds[i].y};
    for (int j=0; j<2; j++) {
      int from=ab[j];
      int through=ab[1-j];
      for (std::set<int>::const_iterator it=angleExcl[through].begin(); it!=angleExcl[through].end(); it++) {
        int to=*it;
        if (to==from) continue;
        if (allExcl[from].count(to)==0) {
          drude_add_exclusion(diheExcl,from,to);
          drude_add_exclusion(allExcl,from,to);
        }
      }
    }
  }

  std::vector< std::set<int> > parentBondExcl=bondExcl;
  std::vector< std::set<int> > parentAngleExcl=angleExcl;

  for (size_t i=0; i<pairs.size(); i++) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    if (allExcl[d].count(p)==0) {
      drude_add_exclusion(bondExcl,d,p);
      drude_add_exclusion(allExcl,d,p);
    }
  }

  for (size_t i=0; i<pairs.size(); i++) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    std::set<int> inherited;
    inherited.insert(parentBondExcl[p].begin(),parentBondExcl[p].end());
    inherited.insert(parentAngleExcl[p].begin(),parentAngleExcl[p].end());
    for (std::set<int>::const_iterator it=inherited.begin(); it!=inherited.end(); it++) {
      int neigh=*it;
      if (neigh==d) continue;
      if (neigh==p) continue;
      if (allExcl[d].count(neigh)==0) {
        drude_add_exclusion(angleExcl,d,neigh);
        drude_add_exclusion(allExcl,d,neigh);
      }
    }
  }

  for (size_t i=0; i<plugin->screenedPairs_tmp.size(); i++) {
    const DrudeScreenedPairPotential &pp=plugin->screenedPairs_tmp[i];
    screenedDrudePairs.insert(drude_ordered_pair_key(pp.idx[0],pp.idx[2]));
    int first[4]={pp.idx[0],pp.idx[0],pp.idx[1],pp.idx[1]};
    int second[4]={pp.idx[2],pp.idx[3],pp.idx[2],pp.idx[3]};
    for (int j=0; j<4; j++) {
      int a=first[j];
      int b=second[j];
      if (a==b) continue;
      if (allExcl[a].count(b)==0) {
        drude_add_exclusion(allExcl,a,b);
      }
    }
  }

  plugin->nbtholePairs_tmp.clear();
  plugin->nbthole14Pairs_tmp.clear();
  for (size_t i=0; i<pairs.size(); i++) {
    int p=pairs[i].y;
    pairIndicesByParentType[struc->atomList[p].atomTypeName].push_back(i);
  }

  std::set<long long> emittedParentPairs;
  for (std::map<TypeName2,real>::const_iterator tt=param->tholePairParameter.begin();
       tt!=param->tholePairParameter.end(); tt++) {
    const std::vector<size_t> &list1=pairIndicesByParentType[tt->first.t[0]];
    const std::vector<size_t> &list2=pairIndicesByParentType[tt->first.t[1]];
    if (list1.empty() || list2.empty()) continue;
    for (size_t ii=0; ii<list1.size(); ii++) {
      for (size_t jj=0; jj<list2.size(); jj++) {
        size_t i=list1[ii];
        size_t j=list2[jj];
        if (i==j) continue;
        if (tt->first.t[0]==tt->first.t[1] && j<=i) continue;
        int d1=pairs[i].x;
        int p1=pairs[i].y;
        int d2=pairs[j].x;
        int p2=pairs[j].y;
        int lo=(p1<p2? p1 : p2);
        int hi=(p1<p2? p2 : p1);
        long long parentKey=drude_ordered_pair_key(p1,p2);
        if (emittedParentPairs.count(parentKey)>0) continue;
        emittedParentPairs.insert(parentKey);
        if (screenedDrudePairs.count(drude_ordered_pair_key(d1,d2))>0) continue;
        bool isDihe=(diheExcl[lo].count(hi)>0);
        bool isExcluded=(allExcl[lo].count(hi)>0);
        real screeningScale=drude_autobuild_screening_scale(system,springKByDrude,d1,p1,d2,p2,true,tt->second);

        if (isDihe) {
          drude_autobuild_add_nbthole_atom_pair(plugin->nbthole14Pairs_tmp,
            struc,p1,p2,screeningScale);
          drude_autobuild_add_nbthole_14_main_subpairs(plugin->nbtholePairs_tmp,
            struc,d1,p1,d2,p2,screeningScale);
        } else if (!isExcluded) {
          drude_autobuild_add_nbthole_all_subpairs(plugin->nbtholePairs_tmp,
            struc,d1,p1,d2,p2,screeningScale);
        }
      }
    }
  }

}

void drude_autobuild(char *line,System *system)
{
  if (!system || !system->drudePlugin) {
    fatal(__FILE__,__LINE__,"drude autobuild requires an initialized drude plugin\n");
  }
  if (!system->structure || system->structure->atomCount<=0) {
    fatal(__FILE__,__LINE__,"drude autobuild requires structure data (load PSF first)\n");
  }
  if (!system->parameters) {
    fatal(__FILE__,__LINE__,"drude autobuild requires parameter data (load PRM first)\n");
  }

  bool buildSprings=false;
  bool buildScreened=false;
  bool buildNbthole=false;
  char mode[MAXLENGTHSTRING];
  io_nexta(line,mode);
  if (strcmp(mode,"")==0 || strcmp(mode,"all")==0) {
    buildSprings=true;
    buildScreened=true;
    buildNbthole=true;
  } else if (strcmp(mode,"springs")==0) {
    buildSprings=true;
  } else if (strcmp(mode,"screened")==0) {
    buildScreened=true;
  } else if (strcmp(mode,"nbthole")==0) {
    buildNbthole=true;
  } else {
    fatal(__FILE__,__LINE__,
      "drude autobuild mode must be one of: all, springs, screened, nbthole (found %s)\n",mode);
  }

  std::vector<int2> detectedPairs;
  if (buildSprings) {
    drude_autobuild_detect_pairs(system,detectedPairs);
    drude_autobuild_build_springs(system,detectedPairs);
  }

  std::vector<int2> effectivePairs=drude_collect_effective_springs(system->drudePlugin);
  if (effectivePairs.empty() && (buildScreened || buildNbthole)) {
    fatal(__FILE__,__LINE__,
      "drude autobuild %s requires drude spring pairs. "
      "Call drude autobuild springs/all or provide manual drude spring directives first.\n",
      (buildScreened && buildNbthole? "all" : (buildScreened? "screened" : "nbthole")));
  }

  if (buildScreened) {
    drude_autobuild_build_screened(system,effectivePairs);
  }
  if (buildNbthole) {
    drude_autobuild_build_nbthole(system,effectivePairs);
  }

  system->drudePlugin->enabled=true;
  fprintf(stdout,
    "DRUDE AUTOBUILD> mode=%s springPairCount=%d screenedPairCount=%d nbtholePairCount=%d nbthole14PairCount=%d\n",
    (strcmp(mode,"")==0? "all" : mode),
    (int)system->drudePlugin->springPairs_tmp.size(),
    (int)system->drudePlugin->screenedPairs_tmp.size(),
    (int)system->drudePlugin->nbtholePairs_tmp.size(),
    (int)system->drudePlugin->nbthole14Pairs_tmp.size());
}
