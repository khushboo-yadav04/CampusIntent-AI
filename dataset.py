import pandas as pd
import os

data = {
    "query": [
        # FEES (20)
        "What is the semester fee?",
        "How can I pay my tuition fees?",
        "What is the hostel fee?",
        "Is there an examination fee?",
        "Can I pay fees in installments?",
        "What is the annual college fee?",
        "How do I get a fee receipt?",
        "What is the late fee penalty?",
        "Are fees refundable?",
        "Where can I pay my college fees?",
        "What is the admission fee?",
        "How much is the tuition fee?",
        "Can I pay fees online?",
        "What is the fee structure for CSE?",
        "When is the last date to pay fees?",
        "How can I check my pending fees?",
        "Is there a scholarship fee waiver?",
        "What are the additional college charges?",
        "Can I get a fee refund?",
        "How do I download my fee receipt?",

        # EXAMS (20)
        "When are semester exams?",
        "Where can I find the exam timetable?",
        "What is the exam registration process?",
        "When will exam results be declared?",
        "How do I apply for revaluation?",
        "What is the passing criteria?",
        "When are practical exams scheduled?",
        "How can I download my hall ticket?",
        "What is the supplementary exam date?",
        "Can I apply for a backlog exam?",
        "Where is the examination department?",
        "How do I check my exam marks?",
        "What is the minimum passing percentage?",
        "When does the end semester exam start?",
        "How can I correct my exam form?",
        "What happens if I miss an exam?",
        "When is the internal assessment?",
        "How do I get my transcript?",
        "What is the exam pattern?",
        "Can I change my exam center?",

        # HOSTEL (15)
        "How can I apply for hostel accommodation?",
        "Is hostel accommodation available?",
        "What are the hostel rules?",
        "When does hostel registration start?",
        "What is the hostel fee?",
        "Are meals included in the hostel?",
        "Can I get a hostel room with a roommate?",
        "What are the hostel timings?",
        "Is WiFi available in the hostel?",
        "How do I report a hostel complaint?",
        "Can I leave the hostel on weekends?",
        "Are separate hostels available for girls?",
        "How can I check hostel room availability?",
        "What documents are required for hostel admission?",
        "Is there a hostel warden?",

        # ADMISSIONS (15)
        "How can I apply for admission?",
        "What is the admission process?",
        "What are the eligibility criteria?",
        "When does admission registration start?",
        "Which documents are required for admission?",
        "What is the last date for admission?",
        "How can I check my admission status?",
        "Are admissions open for engineering?",
        "How do I get an admission confirmation letter?",
        "What entrance exams are accepted?",
        "Can I apply for admission online?",
        "How can I cancel my admission?",
        "What courses are available?",
        "Is direct admission available?",
        "Where is the admission office?",

        # PLACEMENTS (15)
        "What is the placement process?",
        "Which companies visit the campus?",
        "What is the average placement package?",
        "When do campus placements start?",
        "How can I register for placements?",
        "Are internships available?",
        "What is the highest placement package?",
        "How do I prepare for campus interviews?",
        "Which companies recruit CSE students?",
        "Is placement training provided?",
        "What are the placement eligibility criteria?",
        "Where can I find placement statistics?",
        "How can I apply for a company drive?",
        "Are off-campus placement opportunities shared?",
        "Who is the placement coordinator?",

        # SCHOLARSHIPS (15)
        "How can I apply for a scholarship?",
        "What scholarships are available?",
        "What are the scholarship eligibility criteria?",
        "When is the scholarship application deadline?",
        "How can I check scholarship status?",
        "Are merit-based scholarships available?",
        "Is there a scholarship for economically weaker students?",
        "What documents are required for a scholarship?",
        "How much scholarship amount is provided?",
        "Can I renew my scholarship?",
        "How do I apply for a government scholarship?",
        "Are scholarships available for engineering students?",
        "What is the income limit for scholarships?",
        "When will scholarship money be credited?",
        "Where can I contact the scholarship department?"
    ],
    "intent": (
        ["Fees"] * 20 +
        ["Exams"] * 20 +
        ["Hostel"] * 15 +
        ["Admissions"] * 15 +
        ["Placements"] * 15 +
        ["Scholarships"] * 15
    )
}

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)

df.to_csv("data/intents.csv", index=False)

print("Dataset created successfully!")
print("Total queries:", len(df))
print("\nIntent distribution:")
print(df["intent"].value_counts())
print("\nDataset saved to data/intents.csv")