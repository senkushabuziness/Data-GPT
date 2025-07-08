import re
from typing import Tuple

def sanitize_csv(csv_string: str) -> str:
        # Remove JSON-like structures
        csv_string = re.sub(r'\{[^}]*\}', '', csv_string)
        csv_string = re.sub(r'\[.*?\]', '', csv_string)
        # Remove special characters and clean up
        csv_string = re.sub(r'[^\w\s,.\-()%$€£¥]+', '', csv_string)
        csv_string = re.sub(r',,+', ',', csv_string)
        # Remove non-ASCII characters
        csv_string = re.sub(r'[^\x00-\x7F]+', '', csv_string)
        # Clean up lines
        csv_string = '\n'.join(line.strip() for line in csv_string.splitlines() if line.strip())
        return csv_string.strip()

def create_simplified_cleaning_prompt(sample_csv: str) -> str:
        """Create a simplified prompt that's more likely to generate valid JSON"""
        sample_csv = sanitize_csv(sample_csv)
        return f"""You are a financial data analyst. Convert financial CSV data into clean, structured JSON format suitable for database storage. Focus on accuracy, consistency, and completeness.

CSV DATA TO PROCESS:
{sample_csv}

## PROCESSING STEPS

### Step 1: Analyze CSV Structure
1. Identify statement type by scanning for keywords:
   - "Balance Sheet" → balance_sheet
   - "Profit & Loss" OR "Income Statement" → income_statement  
   - "Cash Flow" → cash_flow_statement
   - Multiple statements → multi_statement

2. Detect data orientation:
   - HORIZONTAL: Years/periods as column headers
   - VERTICAL: Years/periods as row entries
   - MIXED: Multiple tables in one file

3. Extract fiscal periods using patterns:
   - FY 'XX → FY_20XX
   - Year X → Year_X
   - 20XX → 20XX

### Step 2: Data Cleaning Rules
Account Names:
- Convert to snake_case: "Net Operating Revenues" → "net_operating_revenues"
- Remove special chars: "Property, plant & equipment" → "property_plant_equipment"
- Standardize terms: "P&E" → "property_plant_equipment"

Numeric Values:
- Remove formatting: "1,20,000" → 120000.0
- Handle negatives: "(5,200)" → -5200.0
- Convert nulls: "-" OR "" → 0.0
- Preserve decimals: "1.5" → 1.5

### Step 3: Key Metric Priorities

Balance Sheet (Priority Order):
1. Assets: cash, accounts_receivable, inventory, total_current_assets, property_plant_equipment, total_assets
2. Liabilities: accounts_payable, short_term_debt, total_current_liabilities, long_term_debt, total_liabilities  
3. Equity: share_capital, retained_earnings, total_equity

Income Statement (Priority Order):
1. Revenue: net_revenue, gross_revenue, operating_revenue
2. Costs: cost_of_goods_sold, operating_expenses, total_expenses
3. Profitability: gross_profit, operating_income, net_income

Cash Flow Statement (Priority Order):
1. Operating: net_cash_from_operations, operating_cash_flow
2. Investing: net_cash_from_investing, capital_expenditures
3. Financing: net_cash_from_financing, debt_issuance, dividends_paid

## REQUIRED OUTPUT FORMAT

Return ONLY this JSON structure with no additional text:

{{
  "statement_type": "balance_sheet|income_statement|cash_flow_statement|multi_statement",
  "company_info": {{
    "name": "extracted_or_unknown",
    "currency": "extracted_or_USD",
    "unit": "extracted_or_actual"
  }},
  "cleaned_data": {{
    "columns": ["fiscal_year", "metric1", "metric2", ...],
    "data": [
      ["FY_2009", 30990.0, 11088.0, ...],
      ["FY_2010", 35119.0, 12693.0, ...]
    ]
  }},
  "schema": {{
    "table_name": "financial_data",
    "columns": {{
      "fiscal_year": "VARCHAR(10) PRIMARY KEY",
      "metric1": "DECIMAL(15,2)",
      "metric2": "DECIMAL(15,2)"
    }}
  }},
  "metadata": {{
    "total_rows": 10,
    "total_columns": 25,
    "fiscal_periods": ["FY_2009", "FY_2010"],
    "data_issues": ["issue1", "issue2"]
  }}
}}

## ERROR HANDLING
- If fiscal periods are unclear, use sequential numbering: Period_1, Period_2
- If statement type is ambiguous, classify as "multi_statement"
- If currency is unclear, default to "USD"
- Missing values should be 0.0, not null

## EXAMPLE TRANSFORMATION

Input CSV Line:
NET OPERATING REVENUES,30,990,35,119,46,542

Output JSON:
{{
  "columns": ["fiscal_year", "net_operating_revenues"],
  "data": [
    ["FY_2009", 30990.0],
    ["FY_2010", 35119.0], 
    ["FY_2011", 46542.0]
  ]
}}

## QUALITY REQUIREMENTS
- All numeric values must be properly formatted decimals
- Column names must be valid database identifiers
- Fiscal years must be consistently formatted
- No missing or null values in the data array
- Schema must match the actual data structure

CRITICAL: Process the provided CSV data above and return the complete JSON transformation following this exact structure. Return ONLY valid JSON with no explanatory text.
"""