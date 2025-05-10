# Stage 1: Install dependencies
FROM node:20-alpine AS deps
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock, pnpm-lock.yaml)
COPY package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm ci --frozen-lockfile

# Stage 2: Build the application
FROM node:20-alpine AS builder
WORKDIR /app

# Copy dependencies from the 'deps' stage
COPY --from=deps /app/node_modules ./node_modules

# Copy the rest of the application code
COPY . .

# Disable Next.js telemetry during the build
ENV NEXT_TELEMETRY_DISABLED 1

# Build the Next.js application
RUN npm run build

# Stage 3: Production image
FROM node:20-alpine AS runner
WORKDIR /app

# Set environment variables for production
ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Create a non-root user and group for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy the standalone Next.js application output from the 'builder' stage
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Copy the public directory from the 'builder' stage
# COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# Set the user to the non-root user
USER nextjs

# Expose the port the app will run on
EXPOSE 3000

# Set the default port for the Next.js application
ENV PORT 3000

# Command to run the Next.js application
CMD ["node", "server.js"]
