
import os
import requests
from loguru import logger
import asyncio

# Target directory
KNOWLEDGE_DIR = "data/knowledge_base"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# List of schemes to seed
SCHEMES = {
    "pmjdy_scheme.txt": {
        "url": "https://www.pmjdy.gov.in/scheme", 
        "content_fallback": """
        Pradhan Mantri Jan-Dhan Yojana (PMJDY)
        Objective: Use of technology to ensure financial inclusion.
        Benefits:
        1. Interest on deposit.
        2. Accidental insurance cover of Rs. 1.00 lac (enhanced to Rs. 2.00 lac for new accounts opened after 28.8.2018).
        3. No minimum balance required.
        4. Life insurance cover of Rs. 30,000/-.
        5. Overdraft facility up to Rs. 10,000/- is available in only one account per household, preferably lady of the household.
        Eligibility: Any individual above 10 years age.
        Documents: Aadhaar, Voter ID, Driving License, PAN Card, Passport, NREGA Card.
        """
    },
    "pmmy_scheme.txt": {
        "url": "https://www.mudra.org.in/",
        "content_fallback": """
        Pradhan Mantri Mudra Yojana (PMMY)
        Objective: To provide funding to the non-corporate, non-farm small/micro enterprises.
        Categories:
        1. Shishu: Loans up to Rs. 50,000/-
        2. Kishore: Loans above Rs. 50,000/- and up to Rs. 5 lakh
        3. Tarun: Loans above Rs. 5 lakh and up to Rs. 10 lakh
        Purpose: Business loan for vendors, traders, shopkeepers and other service sector activities.
        Collateral: No collateral required.
        Processing Fee: Nil for Shishu and Kishore.
        """
    },
    "apy_scheme.txt": {
        "url": "https://npscra.nsdl.co.in/scheme-details.php",
        "content_fallback": """
        Atal Pension Yojana (APY)
        Objective: To provide a defined pension, depending on the contribution, and its period.
        Eligibility:
        - Any Citizen of India.
        - Age between 18 years to 40 years.
        - Should have a savings bank account/post office savings bank account.
        Benefits:
        - Guaranteed minimum pension of Rs. 1000/-, 2000/-, 3000/-, 4000/-, 5000/- per month at the age of 60 years.
        - The pension is guaranteed by the Government of India.
        Contribution: Depends on age of entry and pension amount chosen.
        """
    },
    "rbi_ombudsman.txt": {
        "url": "https://rbi.org.in/Scripts/AboutUsDisplay.aspx?pg=BankingOmbudsman.htm",
        "content_fallback": """
        Reserve Bank - Integrated Ombudsman Scheme, 2021
        Purpose: Cost-free grievance redressal mechanism for customers of regulated entities (Banks, NBFCs, Payment System Participants).
        Grounds for complaint:
        - Non-observance of Fair Practices Code.
        - Levying of charges without prior notice.
        - Delay in payment of inward remittances.
        - Non-adherence to prescribed working hours.
        - Refusal to open accounts without valid reason.
        Procedure:
        - File complaint with the Regulated Entity (Bank/NBFC) first.
        - If rejected or not replied within 30 days, file with RBI Ombudsman via CMS portal (cms.rbi.org.in).
        """
    }
}

async def seed_knowledge():
    """Download or create initial knowledge base files"""
    logger.info("🌱 Seeding Knowledge Base...")
    
    for filename, data in SCHEMES.items():
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        
        if os.path.exists(filepath):
            logger.info(f"✅ {filename} already exists. Skipping.")
            continue
            
        # simulating download by writing fallback content (since actual scraping is complex/blocked)
        # In real scenario, we would use requests.get() and BeautifulSoup
        logger.info(f"💾 Creating {filename}...")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(data["content_fallback"].strip())
            logger.success(f"✅ Created {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to create {filename}: {e}")

    logger.info("✨ Knowledge seeding complete. Ready for ingestion.")

if __name__ == "__main__":
    asyncio.run(seed_knowledge())
