import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Download, Filter, PhoneCall, Gift, Mail, Sparkles } from 'lucide-react'
import crmData from '@/data/crm-data.json'

export function CrmLeaderboard() {
  const [minSpend, setMinSpend] = useState<number>(1000)
  const [minRecency, setMinRecency] = useState<number>(60)

  const { leaderboard } = crmData

  // Filter accounts
  const filtered = leaderboard.filter(
    (item) => item.spend >= minSpend && item.recency >= minRecency
  )

  const totalAtRiskValue = filtered.reduce((acc, curr) => acc + curr.spend, 0)

  const handleExportCsv = () => {
    const headers = [
      'Customer ID',
      'Persona',
      'Churn Probability (%)',
      'Historical Spend (£)',
      'Predicted 90d CLV (£)',
      'Recency (Days)',
      'Priority Score',
      'Recommended CRM Playbook',
    ]
    const rows = filtered.map((row) => [
      row.customer_id,
      `"${row.segment_persona}"`,
      row.churn_prob,
      row.spend,
      row.pred_clv,
      row.recency,
      row.priority_score,
      `"${row.playbook}"`,
    ])
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', 'crm_retention_priority_accounts.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const getPlaybookIcon = (playbook: string) => {
    if (playbook.includes('VIP Concierge')) return <PhoneCall className='size-3 text-rose-600 dark:text-rose-400' />
    if (playbook.includes('Loyalty')) return <Gift className='size-3 text-amber-600 dark:text-amber-400' />
    if (playbook.includes('Win-Back')) return <Mail className='size-3 text-indigo-600 dark:text-indigo-400' />
    return <Sparkles className='size-3 text-emerald-600 dark:text-emerald-400' />
  }

  return (
    <div className='space-y-6'>
      <div className='flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between'>
        <div>
          <h2 className='text-lg font-bold tracking-tight'>Prioritized Retention Outreach Leaderboard</h2>
          <p className='text-xs text-muted-foreground'>
            Target retention playbooks where financial ROI is highest: High-Value Accounts with Elevated Churn Probability.
          </p>
        </div>
        <Button onClick={handleExportCsv} size='sm' className='gap-1.5 self-start sm:self-auto bg-indigo-600 hover:bg-indigo-700 text-white'>
          <Download className='size-3.5' />
          Export Task List for CRM (HubSpot / Salesforce)
        </Button>
      </div>

      {/* Filter Controls */}
      <Card className='border-border/60 shadow-sm'>
        <CardContent className='pt-6'>
          <div className='grid gap-6 sm:grid-cols-2'>
            <div className='space-y-2'>
              <div className='flex justify-between text-xs'>
                <span className='font-medium text-foreground flex items-center gap-1.5'>
                  <Filter className='size-3.5 text-muted-foreground' />
                  Minimum Historical Spend:
                </span>
                <span className='font-bold text-foreground font-mono'>£{minSpend.toLocaleString()}</span>
              </div>
              <input
                type='range'
                min={0}
                max={10000}
                step={250}
                value={minSpend}
                onChange={(e) => setMinSpend(Number(e.target.value))}
                className='w-full accent-indigo-600 cursor-pointer'
              />
            </div>

            <div className='space-y-2'>
              <div className='flex justify-between text-xs'>
                <span className='font-medium text-foreground flex items-center gap-1.5'>
                  <Filter className='size-3.5 text-muted-foreground' />
                  Minimum Inactive Days (Recency):
                </span>
                <span className='font-bold text-foreground font-mono'>{minRecency} Days</span>
              </div>
              <input
                type='range'
                min={0}
                max={365}
                step={10}
                value={minRecency}
                onChange={(e) => setMinRecency(Number(e.target.value))}
                className='w-full accent-indigo-600 cursor-pointer'
              />
            </div>
          </div>

          <div className='mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-border/40 pt-4 text-xs'>
            <div className='flex items-center gap-2'>
              <Badge variant='destructive' className='bg-rose-100 text-rose-800 hover:bg-rose-100 dark:bg-rose-950/60 dark:text-rose-300'>
                {filtered.length} Accounts Identified
              </Badge>
              <span className='text-muted-foreground'>
                At-Risk Portfolio Value: <b className='text-foreground font-mono'>£{totalAtRiskValue.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b>
              </span>
            </div>
            <div className='text-muted-foreground italic text-[11px]'>
              Formula: Priority Score = P(Churn) × Historical Lifetime Spend
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Leaderboard Table */}
      <Card className='border-border/60 shadow-sm overflow-hidden'>
        <div className='overflow-x-auto'>
          <table className='w-full text-left text-xs'>
            <thead className='border-b border-border/60 bg-muted/40 font-semibold text-muted-foreground'>
              <tr>
                <th className='p-3'>Customer ID</th>
                <th className='p-3'>Segment Persona</th>
                <th className='p-3'>Churn Risk</th>
                <th className='p-3'>Historical Spend</th>
                <th className='p-3'>Predicted 90d CLV</th>
                <th className='p-3'>Recency</th>
                <th className='p-3'>Priority Score</th>
                <th className='p-3'>Recommended CRM Playbook</th>
              </tr>
            </thead>
            <tbody className='divide-y divide-border/40'>
              {filtered.slice(0, 25).map((row) => (
                <tr key={row.customer_id} className='hover:bg-muted/30 transition-colors'>
                  <td className='p-3 font-mono font-medium text-foreground'>#{row.customer_id}</td>
                  <td className='p-3 font-medium text-foreground'>{row.segment_persona}</td>
                  <td className='p-3'>
                    <span
                      className={`inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-bold font-mono ${
                        row.churn_prob >= 70
                          ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                          : row.churn_prob >= 50
                          ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300'
                          : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'
                      }`}
                    >
                      {row.churn_prob}%
                    </span>
                  </td>
                  <td className='p-3 font-mono font-semibold text-foreground'>
                    £{row.spend.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                  </td>
                  <td className='p-3 font-mono text-emerald-600 dark:text-emerald-400 font-medium'>
                    £{row.pred_clv.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                  </td>
                  <td className='p-3 font-mono text-muted-foreground'>{row.recency}d</td>
                  <td className='p-3 font-mono font-bold text-foreground'>
                    {row.priority_score.toLocaleString()}
                  </td>
                  <td className='p-3'>
                    <div className='flex items-center gap-1.5 font-medium text-foreground text-[11px]'>
                      {getPlaybookIcon(row.playbook)}
                      {row.playbook}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
