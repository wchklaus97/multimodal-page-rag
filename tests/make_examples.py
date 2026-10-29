"""Regenerate examples/harbour-lane/: python tests/make_examples.py

Harbour Lane Group Limited is a FICTIONAL Hong Kong trading group, used only to give the
demo a realistic multi-entity, multi-department folder structure. Policies are invented
for teaching and are not legal or HR advice.

  harbour-lane/<entity>/<department>/<file>.pdf
"""

from pathlib import Path

from _minipdf import bar_chart, label, make_pdf

ROOT = Path(__file__).parent.parent / "examples" / "harbour-lane"

DOCS = {
    "hk-hq/hr/leave-and-mpf-policy.pdf": [
        {"lines": [
            "Harbour Lane Group Limited - Staff Handbook (fictional)",
            "Section 3: Annual leave",
            "Full-time staff in Hong Kong receive 15 days of annual leave after probation.",
            "Leave applications go to your line manager at least 5 working days ahead.",
            "Unused leave up to 5 days may be carried over to the first quarter of next year.",
        ]},
        {"lines": [
            "Section 4: MPF and benefits",
            "The company enrols all eligible employees in its MPF scheme and pays the",
            "employer contribution in line with the Mandatory Provident Fund Schemes Ordinance.",
            "Medical insurance for staff starts on the first day after probation.",
            "Questions: hr@harbourlane.example",
        ]},
    ],
    "hk-hq/finance/expense-claims.pdf": [
        {"lines": [
            "Finance Procedure FIN-07: Expense claims (fictional)",
            "Submit claims within 30 days with original receipts, in HKD where possible.",
            "Taxi fares above HKD 300 and client meals above HKD 800 per head need",
            "approval from the Finance Manager before the claim is paid.",
        ]},
        {"lines": ["Figure 1"], "ops": bar_chart(
            "Quarterly expense claims by department (HKD thousands)",
            [("HR", 40), ("Sales", 95), ("Ops", 70), ("IT", 55)],
        )},
    ],
    "hk-hq/compliance/pdpo-data-handling.pdf": [
        {"lines": [
            "Compliance Note C-02: Handling personal data (fictional)",
            "Collect only the personal data needed for the stated purpose and tell customers",
            "why it is collected, as required under the Personal Data (Privacy) Ordinance.",
            "Customer lists must not be emailed to personal accounts. Report suspected data",
            "leaks to the Compliance Officer within 24 hours.",
        ]},
    ],
    "sz-subsidiary/operations/supplier-onboarding.pdf": [
        {"lines": [
            "Shenzhen Operations: Supplier onboarding (fictional)",
            "New suppliers submit a business licence copy and bank details to the Shenzhen",
            "procurement team. Orders above RMB 50,000 need co-approval from the Hong Kong",
            "Finance Manager. Payments to suppliers run every second Friday.",
        ]},
    ],
}

if __name__ == "__main__":
    for rel, pages in DOCS.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(make_pdf(pages))
        print("wrote", path.relative_to(ROOT.parent.parent))
