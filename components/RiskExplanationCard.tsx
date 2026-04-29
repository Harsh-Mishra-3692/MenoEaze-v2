"use client"

interface Explanation {
  factor: string
  contributionPercent: number
}

export default function RiskExplanationCard({
  explanation
}: {
  explanation: Explanation[]
}) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h2 className="text-lg font-semibold mb-4 text-purple-700">
        Risk Contribution Analysis
      </h2>

      <div className="space-y-4">
        {explanation.map((item, index) => (
          <div key={index}>
            <div className="flex justify-between text-sm mb-1">
              <span>{item.factor}</span>
              <span>{item.contributionPercent}%</span>
            </div>

            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                style={{ width: `${item.contributionPercent}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}