﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using Microsoft.Extensions.VectorData.ProviderServices;
using Microsoft.Extensions.VectorData.ProviderServices.Filter;

namespace Microsoft.SemanticKernel.Connectors;

#pragma warning disable MEVD9001 // Microsoft.Extensions.VectorData experimental connector-facing APIs

internal abstract class SqlFilterTranslator
{
    private readonly CollectionModel _model;
    private readonly LambdaExpression _lambdaExpression;
    private readonly ParameterExpression _recordParameter;
    protected readonly StringBuilder _sql;

    internal SqlFilterTranslator(
        CollectionModel model,
        LambdaExpression lambdaExpression,
        StringBuilder? sql = null)
    {
        this._model = model;
        this._lambdaExpression = lambdaExpression;
        Debug.Assert(lambdaExpression.Parameters.Count == 1);
        this._recordParameter = lambdaExpression.Parameters[0];
        this._sql = sql ?? new();
    }

    internal StringBuilder Clause => this._sql;

    internal void Translate(bool appendWhere)
    {
        if (appendWhere)
        {
            this._sql.Append("WHERE ");
        }

        var preprocessor = new FilterTranslationPreprocessor { SupportsParameterization = true };
        var preprocessedExpression = preprocessor.Preprocess(this._lambdaExpression.Body);

        this.Translate(preprocessedExpression, isSearchCondition: true);
    }

    protected void Translate(Expression? node, bool isSearchCondition = false)
    {
        switch (node)
        {
            case BinaryExpression binary:
                this.TranslateBinary(binary);
                return;

            case ConstantExpression constant:
                this.TranslateConstant(constant.Value);
                return;

            case QueryParameterExpression { Name: var name, Value: var value }:
                this.TranslateQueryParameter(value);
                return;

            case MemberExpression member:
                this.TranslateMember(member, isSearchCondition);
                return;

            case MethodCallExpression methodCall:
                this.TranslateMethodCall(methodCall, isSearchCondition);
                return;

            case UnaryExpression unary:
                this.TranslateUnary(unary, isSearchCondition);
                return;

            default:
                throw new NotSupportedException("Unsupported NodeType in filter: " + node?.NodeType);
        }
    }

    protected void TranslateBinary(BinaryExpression binary)
    {
        // Special handling for null comparisons
        switch (binary.NodeType)
        {
            case ExpressionType.Equal when IsNull(binary.Right):
                this._sql.Append('(');
                this.Translate(binary.Left);
                this._sql.Append(" IS NULL)");
                return;
            case ExpressionType.NotEqual when IsNull(binary.Right):
                this._sql.Append('(');
                this.Translate(binary.Left);
                this._sql.Append(" IS NOT NULL)");
                return;

            case ExpressionType.Equal when IsNull(binary.Left):
                this._sql.Append('(');
                this.Translate(binary.Right);
                this._sql.Append(" IS NULL)");
                return;
            case ExpressionType.NotEqual when IsNull(binary.Left):
                this._sql.Append('(');
                this.Translate(binary.Right);
                this._sql.Append(" IS NOT NULL)");
                return;
        }

        this._sql.Append('(');
        this.Translate(binary.Left, isSearchCondition: binary.NodeType is ExpressionType.AndAlso or ExpressionType.OrElse);

        this._sql.Append(binary.NodeType switch
        {
            ExpressionType.Equal => " = ",
            ExpressionType.NotEqual => " <> ",

            ExpressionType.GreaterThan => " > ",
            ExpressionType.GreaterThanOrEqual => " >= ",
            ExpressionType.LessThan => " < ",
            ExpressionType.LessThanOrEqual => " <= ",

            ExpressionType.AndAlso => " AND ",
            ExpressionType.OrElse => " OR ",

            _ => throw new NotSupportedException("Unsupported binary expression node type: " + binary.NodeType)
        });

        this.Translate(binary.Right, isSearchCondition: binary.NodeType is ExpressionType.AndAlso or ExpressionType.OrElse);

        this._sql.Append(')');

        static bool IsNull(Expression expression)
            => expression is ConstantExpression { Value: null } or QueryParameterExpression { Value: null };
    }

    protected virtual void TranslateConstant(object? value)
    {
        switch (value)
        {
            case byte b:
                this._sql.Append(b);
                return;
            case short s:
                this._sql.Append(s);
                return;
            case int i:
                this._sql.Append(i);
                return;
            case long l:
                this._sql.Append(l);
                return;

            case float f:
                this._sql.Append(f);
                return;
            case double d:
                this._sql.Append(d);
                return;
            case decimal d:
                this._sql.Append(d);
                return;

            case string untrustedInput:
                // This is the only place where we allow untrusted input to be passed in, so we need to quote and escape it.
                // Luckily for us, values are escaped in the same way for every provider that we support so far.
                this._sql.Append('\'').Append(untrustedInput.Replace("'", "''")).Append('\'');
                return;
            case bool b:
                this._sql.Append(b ? "TRUE" : "FALSE");
                return;
            case Guid g:
                this._sql.Append('\'').Append(g.ToString()).Append('\'');
                return;

            case DateTime dateTime:
            case DateTimeOffset dateTimeOffset:
            case Array:
#if NET8_0_OR_GREATER
            case DateOnly dateOnly:
            case TimeOnly timeOnly:
#endif
                throw new UnreachableException("Database-specific format, needs to be implemented in the provider's derived translator.");

            case null:
                this._sql.Append("NULL");
                return;

            default:
                throw new NotSupportedException("Unsupported constant type: " + value.GetType().Name);
        }
    }

    private void TranslateMember(MemberExpression memberExpression, bool isSearchCondition)
    {
        if (this.TryBindProperty(memberExpression, out var property))
        {
            this.GenerateColumn(property, isSearchCondition);
            return;
        }

        throw new NotSupportedException($"Member access for '{memberExpression.Member.Name}' is unsupported - only member access over the filter parameter are supported");
    }

    protected virtual void GenerateColumn(PropertyModel property, bool isSearchCondition = false)
        // StorageName is considered to be a safe input, we quote and escape it mostly to produce valid SQL.
        => this._sql.Append('"').Append(property.StorageName.Replace("\"", "\"\"")).Append('"');

    protected abstract void TranslateQueryParameter(object? value);

    private void TranslateMethodCall(MethodCallExpression methodCall, bool isSearchCondition = false)
    {
        switch (methodCall)
        {
            // Dictionary access for dynamic mapping (r => r["SomeString"] == "foo")
            case MethodCallExpression when this.TryBindProperty(methodCall, out var property):
                this.GenerateColumn(property, isSearchCondition);
                return;

            // Enumerable.Contains()
            case { Method.Name: nameof(Enumerable.Contains), Arguments: [var source, var item] } contains
                when contains.Method.DeclaringType == typeof(Enumerable):
                this.TranslateContains(source, item);
                return;

            // List.Contains()
            case
            {
                Method:
                {
                    Name: nameof(Enumerable.Contains),
                    DeclaringType: { IsGenericType: true } declaringType
                },
                Object: Expression source,
                Arguments: [var item]
            } when declaringType.GetGenericTypeDefinition() == typeof(List<>):
                this.TranslateContains(source, item);
                return;

            default:
                throw new NotSupportedException($"Unsupported method call: {methodCall.Method.DeclaringType?.Name}.{methodCall.Method.Name}");
        }
    }

    private void TranslateContains(Expression source, Expression item)
    {
        switch (source)
        {
            // Contains over array column (r => r.Strings.Contains("foo"))
            case var _ when this.TryBindProperty(source, out _):
                this.TranslateContainsOverArrayColumn(source, item);
                return;

            // Contains over inline array (r => new[] { "foo", "bar" }.Contains(r.String))
            case NewArrayExpression newArray:
                this.Translate(item);
                this._sql.Append(" IN (");

                var isFirst = true;
                foreach (var element in newArray.Expressions)
                {
                    if (isFirst)
                    {
                        isFirst = false;
                    }
                    else
                    {
                        this._sql.Append(", ");
                    }

                    this.Translate(element);
                }

                this._sql.Append(')');
                return;

            // Contains over captured array (r => arrayLocalVariable.Contains(r.String))
            case QueryParameterExpression { Value: var value }:
                this.TranslateContainsOverParameterizedArray(source, item, value);
                return;

            default:
                throw new NotSupportedException("Unsupported Contains expression");
        }
    }

    protected abstract void TranslateContainsOverArrayColumn(Expression source, Expression item);

    protected abstract void TranslateContainsOverParameterizedArray(Expression source, Expression item, object? value);

    private void TranslateUnary(UnaryExpression unary, bool isSearchCondition)
    {
        switch (unary.NodeType)
        {
            case ExpressionType.Not:
                // Special handling for !(a == b) and !(a != b)
                if (unary.Operand is BinaryExpression { NodeType: ExpressionType.Equal or ExpressionType.NotEqual } binary)
                {
                    this.TranslateBinary(
                        Expression.MakeBinary(
                            binary.NodeType is ExpressionType.Equal ? ExpressionType.NotEqual : ExpressionType.Equal,
                            binary.Left,
                            binary.Right));
                    return;
                }

                this._sql.Append("(NOT ");
                this.Translate(unary.Operand, isSearchCondition);
                this._sql.Append(')');
                return;

            // Handle converting non-nullable to nullable; such nodes are found in e.g. r => r.Int == nullableInt
            case ExpressionType.Convert when Nullable.GetUnderlyingType(unary.Type) == unary.Operand.Type:
                this.Translate(unary.Operand, isSearchCondition);
                return;

            // Handle convert over member access, for dynamic dictionary access (r => (int)r["SomeInt"] == 8)
            case ExpressionType.Convert when this.TryBindProperty(unary.Operand, out var property) && unary.Type == property.Type:
                this.GenerateColumn(property, isSearchCondition);
                return;

            default:
                throw new NotSupportedException("Unsupported unary expression node type: " + unary.NodeType);
        }
    }

    private bool TryBindProperty(Expression expression, [NotNullWhen(true)] out PropertyModel? property)
    {
        var unwrappedExpression = expression;
        while (unwrappedExpression is UnaryExpression { NodeType: ExpressionType.Convert } convert)
        {
            unwrappedExpression = convert.Operand;
        }

        var modelName = unwrappedExpression switch
        {
            // Regular member access for strongly-typed POCO binding (e.g. r => r.SomeInt == 8)
            MemberExpression memberExpression when memberExpression.Expression == this._recordParameter
                => memberExpression.Member.Name,

            // Dictionary lookup for weakly-typed dynamic binding (e.g. r => r["SomeInt"] == 8)
            MethodCallExpression
            {
                Method: { Name: "get_Item", DeclaringType: var declaringType },
                Arguments: [ConstantExpression { Value: string keyName }]
            } methodCall when methodCall.Object == this._recordParameter && declaringType == typeof(Dictionary<string, object?>)
                => keyName,

            _ => null
        };

        if (modelName is null)
        {
            property = null;
            return false;
        }

        if (!this._model.PropertyMap.TryGetValue(modelName, out property))
        {
            throw new InvalidOperationException($"Property name '{modelName}' provided as part of the filter clause is not a valid property name.");
        }

        // Now that we have the property, go over all wrapping Convert nodes again to ensure that they're compatible with the property type
        var unwrappedPropertyType = Nullable.GetUnderlyingType(property.Type) ?? property.Type;
        unwrappedExpression = expression;
        while (unwrappedExpression is UnaryExpression { NodeType: ExpressionType.Convert } convert)
        {
            var convertType = Nullable.GetUnderlyingType(convert.Type) ?? convert.Type;
            if (convertType != unwrappedPropertyType && convertType != typeof(object))
            {
                throw new InvalidCastException($"Property '{property.ModelName}' is being cast to type '{convert.Type.Name}', but its configured type is '{property.Type.Name}'.");
            }

            unwrappedExpression = convert.Operand;
        }

        return true;
    }
}
