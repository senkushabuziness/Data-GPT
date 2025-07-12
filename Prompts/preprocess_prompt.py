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
        return f"""You are a financial data analyst. Transform this financial CSV into clean JSON format for database storage.

CSV Data:
{sample_csv}

ANALYSIS INSTRUCTIONS:
1. DETECT FINANCIAL STATEMENT TYPE:
   - Look for headers like "Balance Sheet", "Profit & Loss", "Income Statement", "Cash Flow"
   - If multiple statements exist, focus on the one with the most complete data
   - Handle both single statement files and multi-statement files

2. IDENTIFY DATA STRUCTURE:
   - HORIZONTAL FORMAT: Fiscal periods as columns (FY '09, FY '10, Year 1, etc.)
   - VERTICAL FORMAT: Fiscal periods as rows with metrics as columns
   - MIXED FORMAT: Multiple statements in one file

3. EXTRACT FISCAL PERIODS: Look for patterns like:
   - FY '09, FY '10, FY '11 (Fiscal Year format)
   - Year 1, Year 2, Year 3 (Simple year format)
   - 2020, 2021, 2022 (Calendar year format)

4. CLEAN ACCOUNT NAMES: Convert to snake_case and remove special characters
   - "NET OPERATING REVENUES" → "net_operating_revenues"
   - "Total Assets" → "total_assets"
   - "Property, plant and equipment" → "property_plant_equipment"
   - "Cash and cash equivalents" → "cash_and_cash_equivalents"

5. CLEAN NUMERIC VALUES: 
   - Remove commas, quotes, currency symbols
   - Handle negatives in parentheses: "(5,200)" → -5200
   - Convert text numbers: "30,990" → 30990.0

6. TRANSPOSE DATA: Make fiscal_year the first column, with financial metrics as other columns

SPECIAL HANDLING:
- If you see "Balance Sheet" or similar headers, treat the first column as account names
- Handle sections like "Current assets:", "Fixed assets:", "Liabilities", "Equity"
- Convert parentheses to negative values: "(5,200)" → -5200
- Handle dashes "-" as zero or null values
- Preserve the hierarchical structure by using descriptive names

REQUIRED OUTPUT FORMAT (JSON only, no additional text):
{{
    "cleaned_data": {{
        "columns": ["fiscal_year", "metric1","metric2",... ]
        "data": [
            ["Year_1", 120000.0, 25000.0, 165000.0, 1300000.0, -5200.0, ....]
        ]
    }},
    "schema": {{
        "columns": {{
            "fiscal_year": "VARCHAR(10) PRIMARY KEY",
             "metric": "VARCHAR(50)",
             ...
        }}
    }},
    "data_quality_report": {{
        "total_rows": 1,
        "total_columns": 16,
        "issues": ["Consolidated multiple fiscal periods into single row", "Converted currency values to decimal format"]
    }}
}}

CRITICAL RULES:
- Return ONLY valid JSON
- AUTO-DETECT the financial statement type from the data
- Include ALL meaningful financial line items as separate columns
- Use descriptive column names in snake_case
- Handle negative values properly (parentheses to negative numbers)
- Create one row per fiscal period found
- Focus on the statement with the most complete data if multiple exist
- Skip metadata/header rows and focus on actual financial data
- Handle both quoted and unquoted numeric values
- Convert fiscal year formats consistently (FY '09 → FY_2009, Year 1 → Year_1)
"""