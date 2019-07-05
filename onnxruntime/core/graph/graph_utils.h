// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace graph_utils {

/** Checks if the operator's type, version, and domain of the given node match the given values. */
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>& versions,
                                       const std::string& domain = kOnnxDomainAlias);

/** Checks if the node has the same operator since version as the given one. */
bool MatchesOpSinceVersion(const Node& node, const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>& versions);

/** Checks if the node has the same op set domain as the given one. */
bool MatchesOpSetDomain(const Node& node, const std::string& domain);

/** Returns true if the execution provider assigned to current node is present in the compatible providers list
    or if the compatible_providers list is empty. */
bool IsSupportedProvider(const Node& node,
                         const std::unordered_set<std::string>& compatible_providers);

/** Checks whether the node has a single input and a single output. The single input can be either the output of
    another node or an initializer, but not an implicit input from a parent subgraph. The single output can be 
    fed to multiple downstream operators, i.e., it can have multiple output edges. */
bool IsSingleInSingleOutNode(const Node& node);

/** Checks if the output at the specified index is input to downstream Nodes. */
bool IsOutputUsed(const Node& node, int index);

/** Returns true if the graph has the given input.*/
bool IsGraphInput(const Graph& graph, const NodeArg* input);

/** returns true if 'name' is an initializer, and is constant and cannot be overridden at runtime. 
@param check_outer_scope If true and the graph is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
*/
bool IsConstantInitializer(const Graph& graph, const std::string& name, bool check_outer_scope = true);

/** returns the initializer's TensorProto if 'name' is an initializer, and is constant and 
cannot be overridden at runtime. If the initializer is not found or is not constant a nullptr is returned.
@param check_outer_scope If true and the graph is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
*/
const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const Graph& graph, const std::string& name,
                                                          bool check_outer_scope = true);

/** Find the initializer called 'original_name' in 'graph', or its ancestors if check_outer_scope is true, 
    and replace wtih 'initializer' in the correct graph.
    Returns true if initializer was replaced. */
//bool ReplaceInitializer(Graph& graph, const std::string& original_name, ONNX_NAMESPACE::TensorProto& new_initializer,
//                        bool check_outer_scope = true);

/** Add a new initializer that will replace an existing one, and return a NodeArg for the new initializer. */
NodeArg& AddReplacementInitializer(Graph& graph, ONNX_NAMESPACE::TensorProto& new_initializer);

/** Checks if the given NodeArg is constant, i.e., it appears in the graph's initializers but not in its inputs. */
bool NodeArgIsConstant(const Graph& graph, const NodeArg& node_arg);

/** Checks if the given node has only constant inputs (initializers) and if so returns them in constant_inputs as they
may come from outer scope. */
bool AllNodeInputsAreConstant(const Graph& graph, const Node& node, InitializedTensorSet& constant_inputs);

/** Gets the name of the incoming NodeArg with the specified index for the given node. */
const std::string& GetNodeInputName(const Node& node, int index);

/** Gets the name of the outgoing NodeArg with the specified index for the given node. */
const std::string& GetNodeOutputName(const Node& node, int index);

/** Returns the attribute of a Node with a given name. */
const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name);

/** Retrieves the values for a repeated attribute of a node and place them to the values vector. */
template <typename T>
bool GetRepeatedNodeAttributeValues(const Node& node,
                                    const std::string& attr_name,
                                    std::vector<T>& values) {
  const auto* attr = graph_utils::GetNodeAttribute(node, attr_name);
  if (attr) {
    values = ONNX_NAMESPACE::RetrieveValues<T>(*attr);
    return true;
  }
  return false;
}

Status ForAllMutableSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);
Status ForAllSubgraphs(const Graph& main_graph, std::function<Status(const Graph&)> func);

/** Check if it will be possible to remove a node.
    We support the removal of the Node as long as the following conditions hold:
    - The node should not produce a graph output.
    - Only one of the outputs is used by downstream operators (but it can have multiple output edges).
    - If the Node has a single incoming node, we can remove the Node and connect its incoming node to its 
      outgoing nodes, if doing so does not clash with any values in any relevant subgraphs. 
    - If the Node output will be replaced by replacement_output_name, we can remove the node if the new output name 
      does not clash with any values in any relevant subgraphs.  
@param replacement_output_name 
  If a new NodeArg will be created to replace the node's output (e.g. creating new initializer) 
  provide the new name that will be used by the NodeArg.
  If nullptr, the node must have one input edge, and the name from that edge will be used in the checks. 
*/
bool CanRemoveNode(const Graph& graph, const Node& node, const std::string* replacement_output_name = nullptr);

/** Removes the given Node from the Graph and keeps Graph consistent by rebuilding needed connections.
See CanRemoveNode for details on when removal is allowed.
@param replacement_output If we are not connecting the single incoming node with the downstream node/s
                          provide the NodeArg that will replace the output from 'node' and be connected 
                          with the downstream node/s.
*/
bool RemoveNodeAndUpdateEdges(Graph& graph, Node& node, NodeArg* replacement_output = nullptr);

/** Removes all output edges from the given Node of the Graph. 
    This should probably be elevated to the Graph API eventually. */
size_t RemoveNodeOutputEdges(Graph& graph, Node& node);

}  // namespace graph_utils

}  // namespace onnxruntime
