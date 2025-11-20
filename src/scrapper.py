from playwright.sync_api import sync_playwright
import re
import pandas as pd

# List of campaign URLs
campaign_urls = [
    "https://www.ketto.org/fundraiser/my-daughter-is-fighting-for-her-life-and-we-need-your-support-to-save-her-1076199",
    "https://www.ketto.org/fundraiser/my-sister-is-suffering-from-myositis-we-need-your-help-to-provide-for-her-treatment",
    "https://www.ketto.org/fundraiser/my-sister-is-suffering-from-myositis-we-need-your-help-to-provide-for-her-treatment",
    "https://www.ketto.org/fundraiser/my-daughter-is-fighting-for-her-life-and-we-need-your-support-to-save-her-1076199",
    "https://www.ketto.org/fundraiser/my-wife-is-suffering-from-brain-haemorrhage-we-need-your-help-to-provide-for-her-treatment-1085189",
    "https://www.ketto.org/fundraiser/my-father-is-suffering-from-liver-cirrhosis-we-need-your-help-to-provide-for-his-treatment-1081473",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-srikanth-balams-treatment",
    "https://www.ketto.org/fundraiser/i-need-your-urgent-support-for-my-ischemic-heart-disease-treatment-1085426",
    "https://www.ketto.org/fundraiser/i-need-your-urgent-support-for-my-kidney-renal-failure-treatment-1085417",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-shivam-sahus-treatment",
    "https://www.ketto.org/fundraiser/my-nephew-is-suffering-from-inherited-bone-marrow-failure-syndromes-ibmfs-we-need-your-help-to-provide-for-his-treatment-1085218",
    "https://www.ketto.org/fundraiser/my-aunt-is-suffering-from-heart-defect-we-need-your-help-to-provide-for-her-treatment-1085214",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-asrar-ansaris-treatment",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-lavanya-sudhir-sherekars-treatment",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-akkati-radhikas-treatment-1084845",
    "https://www.ketto.org/fundraiser/my-sister-is-fighting-for-her-life-and-we-need-your-support-to-save-her-1084687",
    "https://www.ketto.org/fundraiser/my-mother-is-fighting-for-her-life-and-we-need-your-support-to-save-her-1084671",
    "https://www.ketto.org/fundraiser/i-need-your-urgent-support-for-my-oral-cancer-oral-cavity-cancer-treatment-1084648",
    "https://www.ketto.org/fundraiser/i-need-your-urgent-support-for-my-lung-carcinoma-lung-cancer-treatment-1084459",
    "https://www.ketto.org/fundraiser/offer-a-helping-hand-to-support-shukla-mitras-treatment-1084707",
    "https://www.ketto.org/fundraiser/my-baby-twins-battle-for-their-life-and-we-need-your-support-to-save-them-942700?show=endedPage"

]

def clean_amount(text):
    """Removes non-digit characters (like commas, spaces, and currency symbols)."""
    if text and isinstance(text, str):
        return re.sub(r'[^\d]', '', text)
    return "N/A"

def scrape_campaign(page, url):
    """Scrapes a single campaign page and returns structured data."""
    page.goto(url, wait_until="domcontentloaded")
    
    # Wait for amount card
    try:
        page.wait_for_selector('app-amount-card', timeout=10000)
    except:
        print(f"Warning: Could not load amount card for {url}")
    
    # Raised Amount
    raised_amount_xpath = '//div[contains(@class, "amount-card")]/div[1]//app-currency-icon/following-sibling::text()[1]'
    try:
        raised_text_node = page.evaluate(
            'xpath => document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.textContent',
            raised_amount_xpath
        )
        raised_amount = clean_amount(raised_text_node)
    except:
        raised_amount = "N/A"
    
    # Goal Amount
    goal_span_selector = 'span.raised.ng-star-inserted'
    goal_amount_el = page.query_selector(goal_span_selector)
    goal_amount = clean_amount(goal_amount_el.inner_text().strip()) if goal_amount_el else "N/A"
    
    # Supporters
    supporter_span_selector = "div.amount-footer span.raised.supporters.ng-star-inserted > span.numbers"
    supporter_amount_el = page.query_selector(supporter_span_selector)
    supporter_amount = clean_amount(supporter_amount_el.inner_text().strip()) if supporter_amount_el else "N/A"
    
    # Days Left
    parent_selector = 'div.amount-footer span.raised.supporters.ng-star-inserted'
    days_left_count = "N/A"
    for element in page.query_selector_all(parent_selector):
        full_text = element.inner_text().strip()
        if "Days left" in full_text:
            numbers_span = element.query_selector('span.numbers')
            if numbers_span:
                days_left_count = clean_amount(numbers_span.inner_text().strip())
            break
    
    # Campaigner Name
    name_selector = 'mat-card-title.title.campaigner-color.text-capitalize'
    name_el = page.query_selector(name_selector)
    campaigner_name = name_el.inner_text().strip() if name_el else "N/A"
    
    # Full Story
    paragraphs_selector = '#template p'
    paragraph_elements = page.query_selector_all(paragraphs_selector)
    full_story = "\n\n".join([p.inner_text().strip() for p in paragraph_elements if p.inner_text().strip()])
    
    return {
        "url": url,
        "campaigner_name": campaigner_name,
        "raised_amount": raised_amount,
        "goal_amount": goal_amount,
        "supporters": supporter_amount,
        "days_left": days_left_count,
        "story": full_story
    }

# --- Main scraping loop ---
all_campaigns = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    for url in campaign_urls:
        print(f"Scraping: {url}")
        try:
            data = scrape_campaign(page, url)
            all_campaigns.append(data)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    browser.close()

# Save to CSV
df = pd.DataFrame(all_campaigns)
df.to_csv("ketto_campaigns.csv", index=False)
print(f"\nScraped data saved to kettocampaigns.csv with {len(all_campaigns)} records.")
