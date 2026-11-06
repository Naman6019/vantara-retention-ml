import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import crmData from '@/data/crm-data.json'

interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{
    value: number
    name: string
    color: string
  }>
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (active && payload && payload.length) {
    return (
      <div className='rounded-lg border border-border/60 bg-popover/95 p-3 shadow-md backdrop-blur-sm'>
        <p className='text-xs font-semibold text-foreground'>{label}</p>
        <div className='mt-2 flex flex-col gap-1'>
          {payload.map((entry, index) => (
            <div key={`tooltip-${index}`} className='flex items-center justify-between gap-4 text-xs'>
              <span className='flex items-center gap-1.5' style={{ color: entry.color }}>
                <span className='size-2 rounded-full' style={{ backgroundColor: entry.color }} />
                {entry.name}:
              </span>
              <span className='font-mono font-medium text-foreground'>
                £{entry.value.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }
  return null
}

export function CrmRevenueChart() {
  const { monthly_revenue } = crmData

  return (
    <Card className='border-border/60 shadow-sm'>
      <CardHeader>
        <div className='flex items-center justify-between'>
          <div>
            <CardTitle className='text-base font-semibold'>Monthly Revenue Pipeline & Moving Average Forecast</CardTitle>
            <CardDescription className='text-xs'>
              Historical billed transaction volume with 3-month rolling trendline smoothing
            </CardDescription>
          </div>
          <div className='text-xs font-mono text-muted-foreground'>Currency: GBP (£)</div>
        </div>
      </CardHeader>
      <CardContent>
        <div className='h-[340px] w-full'>
          <ResponsiveContainer width='100%' height='100%'>
            <AreaChart data={monthly_revenue} margin={{ top: 10, right: 20, left: 20, bottom: 0 }}>
              <defs>
                <linearGradient id='colorRevenue' x1='0' y1='0' x2='0' y2='1'>
                  <stop offset='5%' stopColor='#4F46E5' stopOpacity={0.25} />
                  <stop offset='95%' stopColor='#4F46E5' stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray='3 3' className='stroke-muted/40' vertical={false} />
              <XAxis dataKey='month' className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} />
              <YAxis
                className='text-xs text-muted-foreground'
                stroke='currentColor'
                tickLine={false}
                tickFormatter={(val) => `£${(val / 1000).toFixed(0)}k`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend verticalAlign='top' align='right' wrapperStyle={{ paddingBottom: 12, fontSize: 12 }} />
              <Area
                type='monotone'
                dataKey='revenue'
                name='Actual Revenue'
                stroke='#4F46E5'
                strokeWidth={2.5}
                fillOpacity={1}
                fill='url(#colorRevenue)'
              />
              <Line
                type='monotone'
                dataKey='sma_3'
                name='3-Month SMA Forecast'
                stroke='#F59E0B'
                strokeWidth={2.5}
                strokeDasharray='4 4'
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
