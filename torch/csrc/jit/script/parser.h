#pragma once
#include "../ir.h"
#include "lexer.h"
#include "tree.h"
#include "tree_views.h"

namespace torch {
namespace jit {
namespace script {

Decl mergeTypesFromTypeComment(
    Decl decl,
    Decl type_annotation_decl,
    bool is_method);

class Parser {
 public:
  explicit Parser(const std::string& str)
      : L(str), shared(sharedParserData()) {}

  Lexer& lexer() {
    return L;
  }

  TreeRef parseFunction(bool is_method);
  TreeRef parseTypeComment(bool parse_full_line = false);

 private:
  template <typename T>
  List<T> parseList(int begin, int sep, int end, T (Parser::*parse)()) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    std::vector<T> elements;
    if (L.cur().kind != end) {
      do {
        elements.push_back((this->*parse)());
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
    return List<T>::create(r, elements);
  }

  Ident parseIdent();
  TreeRef createApply(Expr expr);
  TreeRef parseExpOrExpTuple(int end);
  TreeRef parseBaseExp();
  TreeRef parseOptionalReduction();
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec);
  Expr parseExp() {
    return parseExp(0);
  }
  Expr parseExp(int precedence);
  Const parseConst();
  std::string parseString(const SourceRange& range, const std::string& str);
  StringLiteral parseStringLiteral();
  Expr parseAttributeValue();
  void parseOperatorArguments(TreeList& inputs, TreeList& attributes);
  Expr parseSubscriptExp();
  TreeRef parseSubscript(TreeRef value);
  TreeRef parseParam();
  Param parseBareTypeAnnotation();
  Assign parseAssign(List<Expr> list);
  TreeRef parseStmt();
  TreeRef parseOptionalIdentList();
  TreeRef parseIf(bool expect_if = true);
  TreeRef parseWhile();
  TreeRef parseFor();

  TreeRef parseStatements(bool expect_indent = true);
  Decl parseDecl();

  bool isCharCount(char c, const std::string& str, size_t start, int len);

  // short helpers to create nodes
  TreeRef c(int kind, const SourceRange& range, TreeList&& trees);
  TreeRef makeList(const SourceRange& range, TreeList&& trees);

  Lexer L;
  SharedParserData& shared;
};

struct IRParser {
  explicit IRParser(const std::string& str)
      : L(str), shared(sharedParserData()) {
    std::cout << "ORIGINAL:\n";
    std::cout << str << "\n";
  }

  std::shared_ptr<Graph> parseGraph() {
    graph_ = std::make_shared<Graph>();

    parseGraphDecl();

    return graph_;
  }

  // TODO correctly parse size/stride on Tensors
  size_t parseDimension() {
    L.expect(TK_NUMBER).text();
    return 0;
  }

  // TODO doc
  std::string normalizeIdentifier(const std::string& identifier) {
    bool isNumber =
        std::all_of(identifier.cbegin(), identifier.cend(), [](const char ch) {
          return std::isdigit(ch);
        });
    return isNumber ? "" : identifier;
  }

  void parseGraphInput() {
    const auto idType = parseIdentifierAndType();
    const auto newValue =
        graph_->addInput(normalizeIdentifier(idType.identifier))
            ->setType(idType.type);
    values_[idType.identifier] = newValue;
  }

  struct IdentifierType {
    std::string identifier;
    TypePtr type;
  };

  std::string parseIdentifier() {
    auto t = L.expect('%');
    if (L.cur().kind != TK_IDENT && L.cur().kind != TK_NUMBER) {
      L.expected("valid value identifier");
    }
    const auto name = L.cur().text();
    L.next();
    return name;
  }

  IdentifierType parseIdentifierAndType() {
    const auto name = parseIdentifier();
    L.expect(':');
    const auto type = L.expect(TK_IDENT);
    bool isList = false;
    if (L.cur().kind == '(') {
      // Parse tensor metadata
      std::function<size_t()> parseElement =
          std::bind(&IRParser::parseDimension, this);
      parseList('(', ',', ')', parseElement);
    } else if (L.cur().kind == '[') {
      L.next();
      L.expect(']');
      isList = true;
    }

    // TODO fix this for other types
    TypePtr typePtr;
    if (type.text() == "Double") {
      typePtr = DynamicType::get();
    } else if (type.text() == "Dynamic") {
      typePtr = DynamicType::get();
    } else if (type.text() == "float") {
      typePtr = FloatType::get();
    } else if (type.text() == "int") {
      typePtr = IntType::get();
    } else if (type.text() == "bool") {
      typePtr = BoolType::get();
    } else {
      throw std::runtime_error("Invalid type for value: " + type.text());
    }

    if (isList) {
      typePtr = ListType::create(typePtr);
    }

    return {name, typePtr};
  }

  void parseReturn(int returnKind, Block* block) {
    L.expect(returnKind);
    parseList('(', ',', ')', [this, block]() {
      // register each return value as an output to the graph
      const auto identifier = parseIdentifier();
      const auto value = values_.at(identifier);
      block->registerOutput(value);
    });
    L.expect(TK_NEWLINE);
  }

  void parseInstruction(Block* block) {
    if (L.cur().kind == TK_RETURN || L.cur().kind == TK_ARROW) {
      return parseReturn(L.cur().kind, block);
    }

    // Parse output list
    // %0 : type, %1 : type = aten::opname(%x, %y)
    // ^------------------^
    std::vector<IdentifierType> outputIdTypes;
    parseList(TK_NOTHING, ',', TK_NOTHING, [this, &outputIdTypes]() {
      outputIdTypes.push_back(parseIdentifierAndType());
    });

    // %0 : type, %1 : type = aten::opname(%x, %y)
    //                      ^
    L.expect('=');

    // Parse operator name
    // %0 : type, %1 : type = aten::opname(%x, %y)
    //                        ^----------^
    auto operatorName = L.expect(TK_IDENT).text();

    while (true) {
      // handle namespacing
      if (L.cur().kind != ':') {
        break;
      }
      L.next();
      L.expect(':');
      operatorName += "::";
      operatorName += L.expect(TK_IDENT).text();
    }

    // Construct node
    const auto op = Symbol::fromQualString(operatorName);
    Node* newNode;
    if (op == prim::Constant) {
      // Special casing for prim::Constant nodes
      L.expect('[');
      L.expect(TK_IDENT);
      L.expect('=');
      // TODO constants of other types
      const auto number = L.expect(TK_NUMBER);
      const auto value = std::stoll(number.text());
      {
        WithInsertPoint g(block);
        newNode = graph_->insertConstant(value)->node();
      }
      L.expect(']');

      // `insertConstant` already sets the node output
      JIT_ASSERT(outputIdTypes.size() == 1);
      values_[outputIdTypes.at(0).identifier] = newNode->output();
    } else {
      newNode = block->appendNode(graph_->create(op, /*outputs=*/0));

      // Set node outputs
      for (const auto outputIdType : outputIdTypes) {
        const auto name = normalizeIdentifier(outputIdType.identifier);
        auto output = newNode->addOutput()->setUniqueName(name)->setType(
            outputIdType.type);
        values_[outputIdType.identifier] = output;
      }
    }

    // Parse node inputs
    // %0 : type, %1 : type = aten::opname(%x, %y)
    //                                    ^------^
    parseList('(', ',', ')', [this, newNode]() {
      const auto identifier = parseIdentifier();
      newNode->addInput(values_.at(identifier));
    });

    // TODO Handle scopes
    // %0 : type, %1 : type = aten::opname(%x, %y), scope: Foo/Bar
    //                                            ^--------------^
    if (L.cur().kind == ',') {
      L.next();

      const auto scopeNames = parseScope();
      // scopeNames = foo/bar/baz
      // scopes [root]->foo->bar->qux
      auto graphScope = graph_->scope_root();

      size_t i = 0;
      for (; i < scopeNames.size(); i++) {
        Scope* matchingScope = nullptr;
        for (const auto& scopeChild : graphScope->children()) {
          if (scopeChild->name() == scopeNames[i]) {
            // This was a match, traverse the scope tree
            matchingScope = scopeChild.get();
            break;
          }
        }
        if (matchingScope == nullptr) {
          // We couldn't find a match, break here.
          break;
        }
      }

      // We need to "add" scopes i..end() on the graph.
      at::optional<Symbol> lastMatchingScope;
      if (i > 0) {
        lastMatchingScope = scopeNames[i - 1];
      }

      // Pop scopes until we find the point of divergence (i.e. the last
      // matching scope)
      auto currentScope = graph_->current_scope();
      while (!currentScope->isRoot()) {
        if (lastMatchingScope && currentScope->name() == *lastMatchingScope) {
          break;
        }
        currentScope = currentScope->parent();
        graph_->pop_scope();
      }

      // push scopes from i onward to the graph
      for (; i < scopeNames.size(); i++) {
        graph_->push_scope(scopeNames[i].toUnqualString());
      }
    }

    // Block handling
    if (op == prim::If) {
      L.expect(TK_INDENT);
      const auto blockName = L.expect(TK_IDENT).text();
      auto block = newNode->addBlock();

      // Parse block inputs
      parseList('(', ',', ')', [this, block]() {
        const auto idType = parseIdentifierAndType();
        block->addInput(normalizeIdentifier(idType.identifier))
            ->setType(idType.type);
      });
      L.expect(':');
      parseInstructions(block);
      L.expect(TK_DEDENT);
      return;
    }
    L.expect(TK_NEWLINE);
  }

  std::vector<Symbol> parseScope() {
    const auto scopeKeyword = L.expect(TK_IDENT);
    JIT_ASSERT(scopeKeyword.text() == "scope");
    L.expect(':');
    std::vector<Symbol> scopeNames;

    do {
      const auto scope = L.expect(TK_IDENT);
      const auto scopeName = Symbol::scope(scope.text());
      scopeNames.push_back(scopeName);
    } while (L.nextIf('/'));

    return scopeNames;
  }

  void parseInstructions(Block* block) {
    L.expect(TK_INDENT);
    while (true) {
      parseInstruction(block);
      if (L.nextIf(TK_DEDENT)) {
        break;
      }
    }
  }

  void parseList(int begin, int sep, int end, std::function<void()> parse) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING) {
      L.expect(begin);
    }
    if (L.cur().kind != end) {
      do {
        parse();
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING) {
      L.expect(end);
    }
  }

  void parseGraphDecl() {
    L.expect(TK_GRAPH);
    std::function<void()> parseElement =
        std::bind(&IRParser::parseGraphInput, this);
    parseList('(', ',', ')', parseElement);
    L.expect(':');
    parseInstructions(graph_->block());

    graph_->dump();
    graph_->lint();
  }

 private:
  std::shared_ptr<Graph> graph_;
  std::map<std::string, Value*> values_;
  Lexer L;
  SharedParserData& shared;
};
} // namespace script
} // namespace jit
} // namespace torch
