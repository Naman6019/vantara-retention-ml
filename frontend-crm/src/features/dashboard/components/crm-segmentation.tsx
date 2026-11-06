import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import crmData from '@/data/crm-data.json'

export function CrmSegmentation() {
  const { personas } = crmData

  const personaMeta: Record<string, { color: string; badge: string; desc: string }> = {
    'High-Value Champions (VIP)': {
      color: '#4F46E5',
      badge: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/60 dark:text-indigo-300',
      desc: 'Top revenue contributors with frequent repeat transactions and lowest churn rate.',
    },
    'Loyal / Steady Core': {
      color: '#06B6D4',
      badge: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-950/60 dark:text-cyan-300',
      desc: 'Consistent mid-tier buyers who form the revenue backbone of the business.',
    },
    'At-Risk / Lapsed High-Value': {
      color: '#F59E0B',
      badge: 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300',
      desc: 'Historically lucrative customers exhibiting prolonged inactivity and impending churn.',
    },
    'Dormant / Low-Engagement': {
      color: '#EF4444',
      badge: 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300',
      desc: 'Accounts with minimal order frequency and high lapse rates requiring automated win-back.',
    },
  }

  // Chart data formatted
  const chartData = personas.map((p) => ({
    name: p.persona.replace(' (VIP)', '').replace(' / Lapsed High-Value', '').replace(' / Low-Engagement', ''),
    fullName: p.persona,
    avgSpend: p.avg_spend,
    churnRate: p.churn_rate,
    accounts: p.count,
  }))

  return (
    <div className='space-y-6'>
      <div>
        <h2 className='text-lg font-bold tracking-tight'>Behavioral Customer Personas (K-Means Clustering)</h2>
        <p className='text-xs text-muted-foreground'>
          Actionable customer segmentation derived from RFM economics, purchase rhythm, and engagement vectors.
        </p>
      </div>

      {/* Persona Cards Grid */}
      <div className='grid gap-4 sm:grid-cols-2 lg:grid-cols-3'>
        {personas.map((p) => {
          const meta = personaMeta[p.persona] || {
            color: '#6366F1',
            badge: 'bg-slate-100 text-slate-800',
            desc: 'Customer behavioral cluster.',
          }
          return (
            <Card key={p.persona} className='border-border/60 shadow-sm'>
              <CardHeader className='pb-2'>
                <div className='flex items-start justify-between gap-2'>
                  <CardTitle className='text-sm font-semibold leading-tight'>{p.persona}</CardTitle>
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold shrink-0 ${meta.badge}`}>
                    {p.count.toLocaleString()} accounts
                  </span>
                </div>
                <CardDescription className='text-xs mt-1'>{meta.desc}</CardDescription>
              </CardHeader>
              <CardContent className='space-y-2 pt-2 text-xs'>
                <div className='flex justify-between py-1 border-b border-border/40'>
                  <span className='text-muted-foreground'>Avg Spend:</span>
                  <span className='font-semibold text-foreground font-mono'>
                    £{p.avg_spend.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div className='flex justify-between py-1 border-b border-border/40'>
                  <span className='text-muted-foreground'>Avg Recency:</span>
                  <span className='font-semibold text-foreground font-mono'>{p.avg_recency} days</span>
                </div>
                <div className='flex justify-between py-1 border-b border-border/40'>
                  <span className='text-muted-foreground'>Order Frequency:</span>
                  <span className='font-semibold text-foreground font-mono'>{p.avg_frequency} orders</span>
                </div>
                <div className='flex justify-between py-1'>
                  <span className='text-muted-foreground'>Segment Churn Rate:</span>
                  <span className='font-bold text-rose-600 dark:text-rose-400 font-mono'>{p.churn_rate}%</span>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Visual Persona Comparison Chart */}
      <div className='grid gap-6 md:grid-cols-2'>
        <Card className='border-border/60 shadow-sm'>
          <CardHeader>
            <CardTitle className='text-sm font-semibold'>Average Spend per Account (£)</CardTitle>
            <CardDescription className='text-xs'>Economic value breakdown by customer persona</CardDescription>
          </CardHeader>
          <CardContent>
            <div className='h-[260px] w-full'>
              <ResponsiveContainer width='100%' height='100%'>
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray='3 3' className='stroke-muted/40' vertical={false} />
                  <XAxis dataKey='name' className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} />
                  <YAxis className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} tickFormatter={(val) => `£${val}`} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className='rounded-lg border border-border/60 bg-popover/95 p-2 shadow-md text-xs'>
                            <p className='font-semibold text-foreground'>{label}</p>
                            <p className='font-mono font-bold text-indigo-600 dark:text-indigo-400'>
                              Avg Spend: £{Number(payload[0]?.value).toLocaleString()}
                            </p>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Bar isAnimationActive={false} dataKey='avgSpend' name='Avg Spend (£)' fill='#4F46E5' radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className='border-border/60 shadow-sm'>
          <CardHeader>
            <CardTitle className='text-sm font-semibold'>Observed Churn Rate (%)</CardTitle>
            <CardDescription className='text-xs'>Risk propensity by behavioral segment</CardDescription>
          </CardHeader>
          <CardContent>
            <div className='h-[260px] w-full'>
              <ResponsiveContainer width='100%' height='100%'>
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray='3 3' className='stroke-muted/40' vertical={false} />
                  <XAxis dataKey='name' className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} />
                  <YAxis className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} tickFormatter={(val) => `${val}%`} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className='rounded-lg border border-border/60 bg-popover/95 p-2 shadow-md text-xs'>
                            <p className='font-semibold text-foreground'>{label}</p>
                            <p className='font-mono font-bold text-rose-600 dark:text-rose-400'>
                              Churn Rate: {payload[0]?.value}%
                            </p>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Bar isAnimationActive={false} dataKey='churnRate' name='Churn Rate (%)' fill='#EF4444' radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
